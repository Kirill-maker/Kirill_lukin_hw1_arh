#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
QUICK="${1:-}"

ARCH="$(uname -m)"
if [ "${ARCH}" != "x86_64" ]; then
  echo "Процессор ${ARCH}, а проекту нужен x86_64 с AVX2."
  echo "На Mac с Apple M1–M4 запустите проект в GitHub Codespaces, см. README."
  exit 1
fi

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
if [ "$(uname -s)" = "Darwin" ]; then
  JOBS="$(sysctl -n hw.ncpu)"
  if sysctl -n machdep.cpu.leaf7_features 2>/dev/null | grep -q AVX512F; then AVX512=ON; else AVX512=OFF; fi
  if command -v brew >/dev/null && brew --prefix libomp >/dev/null 2>&1; then
    CMAKE_ARGS+=(-DOpenMP_ROOT="$(brew --prefix libomp)")
  fi
else
  JOBS="$(nproc)"
  if grep -q avx512f /proc/cpuinfo; then AVX512=ON; else AVX512=OFF; fi
fi
CMAKE_ARGS+=(-DHW_AVX512="${AVX512}")
echo "AVX-512: ${AVX512}, ядер: ${JOBS}"

cmake -S . -B build "${CMAKE_ARGS[@]}"
cmake --build build -j "${JOBS}"

cmake -S . -B build-autovec "${CMAKE_ARGS[@]}" -DHW_SCALAR_AUTOVEC=ON
cmake --build build-autovec -j "${JOBS}" --target hw_bench

./build/hw_tests

mkdir -p results
./build/hw_bench ${QUICK} --csv results/run1.csv
./build/hw_bench ${QUICK} --csv results/run2.csv
./build-autovec/hw_bench ${QUICK} --backend scalar --csv results/autovec.csv

python3 scripts/make_tables.py results/run1.csv results/run2.csv \
        --autovec results/autovec.csv > results/tables.md
echo "Таблицы: results/tables.md"
