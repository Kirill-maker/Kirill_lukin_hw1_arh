#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace t {

struct Case {
    const char* name;
    void (*fn)();
};

inline std::vector<Case>& cases() {
    static std::vector<Case> all;
    return all;
}

inline std::string& context() {
    static std::string text;
    return text;
}

inline int& failures() {
    static int count = 0;
    return count;
}

struct Register {
    Register(const char* name, void (*fn)()) { cases().push_back({name, fn}); }
};

inline void fail(const char* file, int line, const char* expr) {
    if (++failures() <= 5) {
        std::printf("    %s:%d: CHECK(%s) failed [%s]\n", file, line, expr, context().c_str());
    }
}

}

#define TEST(name)                                      \
    static void name();                                 \
    static const t::Register register_##name(#name, &name); \
    static void name()

#define CHECK(cond)                                      \
    do {                                                 \
        if (!(cond)) t::fail(__FILE__, __LINE__, #cond); \
    } while (0)
