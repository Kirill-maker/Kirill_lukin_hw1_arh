#!/usr/bin/env python3

import argparse
import csv

ARRAY_OPS = ["add", "sub", "mul", "div", "sum", "mean", "silu", "gelu"]
TYPES = ["int32", "float32", "float64"]
BACKEND_TITLES = {"scalar": "Scalar", "avx2": "AVX2", "avx512": "AVX-512"}


def load(paths):
    data, threads = {}, 1
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["op"], row["type"], int(row["n"]), row["backend"], row["exec"])
                value = float(row["ns_per_element"])
                data[key] = min(value, data.get(key, value))
                threads = max(threads, int(row["threads"]))
    return data, threads


def size_label(n, matrix):
    if matrix:
        return f"{n}×{n}"
    power = n.bit_length() - 1
    return f"2^{power}" if n == 1 << power else str(n)


def table(title, data, ops, cell):
    matrix = ops == ["matmul"]
    sizes = sorted({n for (op, _, n, _, _) in data if op in ops})
    rows = []
    for op in ops:
        for typ in TYPES:
            cells = [cell(op, typ, n) for n in sizes]
            if any(c is not None for c in cells):
                rows.append([op, typ] + [c or "" for c in cells])
    if not rows:
        return

    header = ["операция", "тип"] + [size_label(n, matrix) for n in sizes]
    print(f"**{title}**\n")
    print("| " + " | ".join(header) + " |")
    print("|---|---|" + "---:|" * len(sizes))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def time_cell(data, backend, mode):
    def cell(op, typ, n):
        value = data.get((op, typ, n, backend, mode))
        return None if value is None else f"{value:.3f}"

    return cell


def ratio_cell(numerator, denominator):
    def cell(op, typ, n):
        slow, fast = numerator(op, typ, n), denominator(op, typ, n)
        return None if slow is None or fast is None else f"{slow / fast:.2f}"

    return cell


def main():
    parser = argparse.ArgumentParser(description="Строит таблицы для отчёта из CSV, который пишет hw_bench.")
    parser.add_argument("results", nargs="+")
    parser.add_argument("--autovec", nargs="+")
    args = parser.parse_args()

    data, threads = load(args.results)
    backends = [b for b in BACKEND_TITLES if any(k[3] == b for k in data)]
    simd = [b for b in backends if b != "scalar"]
    has_parallel = any(k[4] == "par" for k in data)
    autovec = load(args.autovec)[0] if args.autovec else None

    def at(source, backend, mode):
        return lambda op, typ, n: source.get((op, typ, n, backend, mode))

    for ops, what in ((ARRAY_OPS, "массивы"), (["matmul"], "матричное умножение")):
        unit = "нс на одно умножение со сложением" if ops == ["matmul"] else "нс на элемент"
        table(f"Scalar, {unit} ({what})", data, ops, time_cell(data, "scalar", "seq"))
        for b in simd:
            table(f"{BACKEND_TITLES[b]}: во сколько раз быстрее Scalar ({what})", data, ops,
                  ratio_cell(at(data, "scalar", "seq"), at(data, b, "seq")))
        if autovec:
            table(f"Scalar с автовекторизацией: во сколько раз быстрее Scalar без неё ({what})",
                  data, ops, ratio_cell(at(data, "scalar", "seq"), at(autovec, "scalar", "seq")))
        if has_parallel:
            for b in backends:
                table(f"{BACKEND_TITLES[b]}, {threads} потоков: во сколько раз быстрее "
                      f"одного потока ({what})", data, ops,
                      ratio_cell(at(data, b, "seq"), at(data, b, "par")))


if __name__ == "__main__":
    main()
