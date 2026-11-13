cmake_minimum_required(VERSION 3.16)
project(simd_hw1 LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  message(FATAL_ERROR
    "simd_hw1 needs an x86_64 CPU with AVX2 and FMA, "
    "but this machine is ${CMAKE_SYSTEM_PROCESSOR}. "
    "On an Apple Silicon Mac use an x86_64 Linux machine, e.g. GitHub Codespaces (see README).")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HW_AVX512 "Build the optional AVX-512 backend (the CPU must support AVX-512F)" OFF)
option(HW_SCALAR_AUTOVEC "Let the compiler auto-vectorize the scalar backend" OFF)

add_library(hw INTERFACE)
target_include_directories(hw INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_compile_options(hw INTERFACE -mavx2 -mfma -ffp-contract=off)

if(HW_AVX512)
  target_compile_options(hw INTERFACE -mavx512f)
  target_compile_definitions(hw INTERFACE HW_AVX512)
endif()

if(NOT HW_SCALAR_AUTOVEC)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(hw INTERFACE -fno-tree-vectorize)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(hw INTERFACE -fno-vectorize -fno-slp-vectorize)
  endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(hw INTERFACE OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found: hw::par will run sequentially")
endif()

add_executable(hw_tests
  tests/main.cpp
  tests/test_arith.cpp
  tests/test_reduce.cpp
  tests/test_activation.cpp
  tests/test_matmul.cpp
  tests/test_parallel.cpp
)
target_link_libraries(hw_tests PRIVATE hw)
target_compile_options(hw_tests PRIVATE -Wall -Wextra -Wpedantic)

add_executable(hw_bench bench/bench.cpp)
target_link_libraries(hw_bench PRIVATE hw)
target_compile_options(hw_bench PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME hw_tests COMMAND hw_tests)
