import os
from datetime import datetime

import typing

from typing.io import IO
from utils import native_size_map, next_power_of_2
from bitonic_isa import BitonicISA


class AVX2BitonicISA(BitonicISA):
    def __init__(self, type):
        self.vector_size_in_bytes = 32

        self.type = type

        self.bitonic_size_map = {}

        for t, s in native_size_map.items():
            self.bitonic_size_map[t] = int(self.vector_size_in_bytes / s)

        self.bitonic_type_map = {
            "int16_t":  "__m256i",
            "uint16_t": "__m256i",
            "int32_t":  "__m256i",
            "uint32_t": "__m256i",
            "float":    "__m256",
            "int64_t":  "__m256i",
            "uint64_t": "__m256i",
            "double":   "__m256d",
        }

    def max_bitonic_sort_vectors(self):
        return 4

    def vector_size(self):
        return self.bitonic_size_map[self.type]

    def vector_type(self):
        return self.bitonic_type_map[self.type]

    @classmethod
    def supported_types(cls):
        return native_size_map.keys()

    def t2d(self, v: str):
        t = self.type
        if t == "double":
            return v
        elif t == "float":
            return f"s2d({v})"
        return f"i2d({v})"

    def i2t(self, v: str):
        t = self.type
        if t == "double":
            return f"i2d({v})"
        elif t == "float":
            return f"i2s({v})"
        return v

    def d2t(self, v: str):
        t = self.type
        if t == "double":
            return v
        elif t == "float":
            return f"d2s({v})"
        return f"d2i({v})"

    def t2i(self, v: str):
        t = self.type
        if t == "double":
            return f"d2i({v})"
        elif t == "float":
            return f"s2i({v})"
        return v

    def generate_param_list(self, start: int, numParams: int):
        return str.join(", ", list(map(lambda p: f"d{p:02d}", range(start, start + numParams))))

    def generate_param_def_list(self, numParams: int):
        t = self.type
        return str.join(", ", list(map(lambda p: f"TV& d{p:02d}", range(1, numParams + 1))))

    def generate_shuffle_X1(self, v: str):
        size = self.vector_size()
        if size == 16:
            return self.i2t(f"_mm256_shuffle_epi8({self.t2i(v)}, x1)")
        elif size == 8:
            return self.i2t(f"_mm256_shuffle_epi32({self.t2i(v)}, 0b10'11'00'01)")
        elif size == 4:
            return self.d2t(f"_mm256_shuffle_pd({self.t2d(v)}, {self.t2d(v)}, 0b0'1'0'1)")
        raise Exception("WTF")


    def generate_shuffle_X2(self, v: str):
        size = self.vector_size()
        if size == 16:
            return self.i2t(f"_mm256_shuffle_epi32({self.t2i(v)}, 0b10'11'00'01)")
        if size == 8:
            return self.i2t(f"_mm256_shuffle_epi32({self.t2i(v)}, 0b01'00'11'10)")
        elif size == 4:
            return self.d2t(f"_mm256_permute4x64_pd({self.t2d(v)}, 0b01'00'11'10)")
        raise Exception("WTF")


    def generate_shuffle_X4(self, v: str):
        size = self.vector_size()
        if size == 16:
            return self.i2t(f"_mm256_shuffle_epi32({self.t2i(v)}, 0b01'00'11'10)")
        if size == 8:
            return self.d2t(f"_mm256_permute4x64_pd({self.t2d(v)}, 0b01'00'11'10)")
        elif size == 4:
            return self.d2t(f"_mm256_permute4x64_pd({self.t2d(v)}, 0b01'00'11'10)")
        raise Exception("WTF")

    def generate_shuffle_X8(self, v: str):
        size = self.vector_size()
        if size == 16:
            return self.d2t(f"_mm256_permute4x64_pd({self.t2d(v)}, 0b01'00'11'10)")
        raise Exception("WTF")

    def generate_blend_mask(self, blend: int, width: int, asc: bool):
        size = self.vector_size()
        mask = 0
        while size > 0:
            mask = mask <<  width | blend
            size = size - width

        if not asc:
            mask = ~mask

        mask = mask & ((1 << self.vector_size()) - 1)
        return mask

    def generate_blend(self, v1: str, v2: str, blend: int, width: int, asc: bool):
        size = self.vector_size()
        mask = self.generate_blend_mask(blend, width, asc)
        if size == 16:
            return self.i2t(f"_mm256_blend_epi16({self.t2i(v1)}, {self.t2i(v2)}, 0b{mask:016b})")
        if size == 8:
            return self.i2t(f"_mm256_blend_epi32({self.t2i(v1)}, {self.t2i(v2)}, 0b{mask:08b})")
        elif size == 4:
            return self.d2t(f"_mm256_blend_pd({self.t2d(v1)}, {self.t2d(v2)}, 0b{mask:08b})")
        raise Exception("WTF")

    def generate_reverse(self, v: str):
        size = self.vector_size()
        if size == 8:
            v = f"_mm256_shuffle_epi32({self.t2i(v)}, 0b00'01'10'11)"
            return self.d2t(f"_mm256_permute4x64_pd(i2d({v}), 0b01'00'11'10)")
        elif size == 4:
            return self.d2t(f"_mm256_permute4x64_pd({self.t2d(v)}, 0b00'01'10'11)")
        raise Exception("WTF")

    def crappity_crap_crap(self, v1: str, v2: str):
        t = self.type
        if t == "int64_t":
            return f"cmp = _mm256_cmpgt_epi64({v1}, {v2});"
        elif t == "uint64_t":
            return f"cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(topBit, {v1}), _mm256_xor_si256(topBit, {v2}));"

        return "<<<REMOVE_ME>>>"

    def generate_min(self, v1: str, v2: str):
        t = self.type
        if t == "int16_t":
            return f"_mm256_min_epi16({v1}, {v2})"
        elif t == "uint16_t":
            return f"_mm256_min_epu16({v1}, {v2})"
        elif t == "int32_t":
            return f"_mm256_min_epi32({v1}, {v2})"
        elif t == "uint32_t":
            return f"_mm256_min_epu32({v1}, {v2})"
        elif t == "float":
            return f"_mm256_min_ps({v1}, {v2})"
        elif t == "int64_t":
            return self.d2t(f"_mm256_blendv_pd({self.t2d(v1)}, {self.t2d(v2)}, i2d(cmp))")
        elif t == "uint64_t":
            return self.d2t(f"_mm256_blendv_pd({self.t2d(v1)}, {self.t2d(v2)}, i2d(cmp))")
        elif t == "double":
            return f"_mm256_min_pd({v1}, {v2})"
        raise Exception("WTF")

    def generate_max(self, v1: str, v2: str):
        t = self.type
        if t == "int16_t":
            return f"_mm256_max_epi16({v1}, {v2})"
        elif t == "uint16_t":
            return f"_mm256_max_epu16({v1}, {v2})"
        elif t == "int32_t":
            return f"_mm256_max_epi32({v1}, {v2})"
        elif t == "uint32_t":
            return f"_mm256_max_epu32({v1}, {v2})"
        elif t == "float":
            return f"_mm256_max_ps({v1}, {v2})"
        elif t == "int64_t":
            return self.d2t(f"_mm256_blendv_pd({self.t2d(v2)}, {self.t2d(v1)}, i2d(cmp))")
        elif t == "uint64_t":
            return self.d2t(f"_mm256_blendv_pd({self.t2d(v2)}, {self.t2d(v1)}, i2d(cmp))")
        elif t == "double":
            return f"_mm256_max_pd({v1}, {v2})"
        raise Exception("WTF")

    def get_load_intrinsic(self, v: str, offset: int):
        t = self.type
        if t == "double":
            return f"_mm256_loadu_pd(({t} const *) ((__m256d const *) {v} + {offset}))"
        if t == "float":
            return f"_mm256_loadu_ps(({t} const *) ((__m256 const *) {v} + {offset}))"
        return f"_mm256_lddqu_si256((__m256i const *) {v} + {offset});"

    def get_mask_load_intrinsic(self, v: str, offset: int, mask):
        t = self.type

        if self.vector_size() == 4:
            int_suffix = "epi64"
            max_value = f"_mm256_andnot_si256({mask}, _mm256_set1_epi64x(MAX))"
        elif self.vector_size() == 8:
            int_suffix = "epi32"
            max_value = f"_mm256_andnot_si256({mask}, _mm256_set1_epi32(MAX))"

        if t == "double":
            max_value = f"_mm256_andnot_pd(i2d(mask), _mm256_set1_pd(MAX))"
            load = f"_mm256_maskload_pd(({t} const *) ((__m256d const *) {v} + {offset}), {mask})"
            return f"_mm256_or_pd({load}, {max_value})"
        if t == "float":
            max_value = f"_mm256_andnot_ps(i2s(mask), _mm256_set1_ps(MAX))"
            load = f"_mm256_maskload_ps(({t} const *) ((__m256 const *) {v} + {offset}), {mask})"
            return f"_mm256_or_ps({load}, {max_value})"


        if t == "int64_t" or t == "uint64_t":
            it = "long long"
        else:
            it = t[1:] if t[0] == 'u' else t

        load = f"_mm256_maskload_{int_suffix}(({it} const *) ((__m256i const *) {v} + {offset}), {mask})"
        return f"_mm256_or_si256({load}, {max_value})"

    def get_store_intrinsic(self, ptr, offset, value):
        t = self.type
        if t == "double":
            return f"_mm256_storeu_pd(({t} *) ((__m256d *)  {ptr} + {offset}), {value})"
        if t == "float":
            return f"_mm256_storeu_ps(({t} *) ((__m256 *)  {ptr} + {offset}), {value})"
        return f"_mm256_storeu_si256((__m256i *) {ptr} + {offset}, {value})"

    def get_mask_store_intrinsic(self, ptr, offset, value, mask):
        t = self.type

        if self.vector_size() == 4:
            int_suffix = "epi64"
        elif self.vector_size() == 8:
            int_suffix = "epi32"

        if t == "double":
            return f"_mm256_maskstore_pd(({t} *) ((__m256d *)  {ptr} + {offset}), {mask}, {value})"
        if t == "float":
            return f"_mm256_maskstore_ps(({t} *) ((__m256 *)  {ptr} + {offset}), {mask}, {value})"

        if t == "int64_t" or t == "uint64_t":
            it = "long long"
        else:
            it = t[1:] if t[0] == 'u' else t;
        return f"_mm256_maskstore_{int_suffix}(({it} *) ((__m256i *) {ptr} + {offset}), {mask}, {value})"

    def clean_print(self, s: str, file: IO):
        clean_lines = str.join("\n", [l for l in s.splitlines() if not "<<<REMOVE_ME>>>" in l])
        print(clean_lines, file=file)

    def autogenerated_blabber(self):
        return f"""/////////////////////////////////////////////////////////////////////////////
////
// This file was auto-generated by a tool at {datetime.now().strftime("%F %H:%M:%S")}
//
// It is recommended you DO NOT directly edit this file but instead edit
// the code-generator that generated this source file instead.
/////////////////////////////////////////////////////////////////////////////"""

    def generate_prologue(self, f: IO):
        t = self.type
        s = f"""{self.autogenerated_blabber()}

#ifndef BITONIC_MACHINE_AVX2_{t.upper()}_H
#define BITONIC_MACHINE_AVX2_{t.upper()}_H

#ifdef __GNUC__
#ifdef __clang__
#pragma clang attribute push (__attribute__((target("avx2"))), apply_to = any(function))
#else
#pragma GCC push_options
#pragma GCC target("avx2")
#endif
#endif

#include <cassert>
#include <limits>
#include <immintrin.h>
#include "bitonic_machine.h"

#define i2d _mm256_castsi256_pd
#define d2i _mm256_castpd_si256
#define i2s _mm256_castsi256_ps
#define s2i _mm256_castps_si256
#define s2d _mm256_castps_pd
#define d2s _mm256_castpd_ps

namespace vxsort {{
namespace smallsort {{

template<> struct bitonic_machine<{t}, AVX2> {{
    static const int N = {self.vector_size()};
    static constexpr {t} MAX = std::numeric_limits<{t}>::max();
public:
    typedef {self.vector_type()} TV;
    typedef {self.vector_type()} TMASK;

"""
        print(s, file=f)

    def generate_epilogue(self, f: IO):
        s = f"""
}};
}}
}}

#undef i2d
#undef d2i
#undef i2s
#undef s2i
#undef s2d
#undef d2s

#ifdef __GNUC__
#ifdef __clang__
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif
#endif
#endif
    """
        print(s, file=f)

    def generate_1v_basic_sorters(self, f: IO, asc: bool):
        g = self
        type = self.type
        maybe_cmp = lambda: ", cmp" if (type == "int64_t" or type == "uint64_t") else ""
        maybe_topbit = lambda: f"\n        TV topBit = _mm256_set1_epi64x(1LLU << 63);" if (type == "uint64_t") else ""
        maybe_x1_epi16 = lambda: f"\n        const TV x1 = _mm256_set1_epi64x(0xE'F'C'D'A'B'8'9'6'7'4'5'2'3'0'1);" if (
                type == "uint16_t" or type == "int16_t") else ""
        sfx = "ascending" if asc else "descending"

        s = f"""    static INLINE void sort_01v_{sfx}({g.generate_param_def_list(1)}) {{
            TV min, max, s{maybe_cmp()};{maybe_topbit()}{maybe_x1_epi16()}

            s = {g.generate_shuffle_X1("d01")};
            {g.crappity_crap_crap("s", "d01")}
            min = {g.generate_min("s", "d01")};
            max = {g.generate_max("s", "d01")};
            d01 = {g.generate_blend("min", "max", 0b0110, 4, asc)};

            s = {g.generate_shuffle_X2("d01")};
            {g.crappity_crap_crap("s", "d01")}
            min = {g.generate_min("s", "d01")};
            max = {g.generate_max("s", "d01")};
            d01 = {g.generate_blend("min", "max", 0b00111100, 8, asc)};

            s = {g.generate_shuffle_X1("d01")};
            {g.crappity_crap_crap("s", "d01")}
            min = {g.generate_min("s", "d01")};
            max = {g.generate_max("s", "d01")};
            d01 = {g.generate_blend("min", "max", 0b01011010, 8, asc)};"""

        self.clean_print(s, file=f)

        if g.vector_size() == 8:
            s = f"""
            s = {g.generate_shuffle_X4("d01")};
            min = {g.generate_min("s", "d01")};
            max = {g.generate_max("s", "d01")};
            d01 = {g.generate_blend("min", "max", 0b11110000, 8, asc)};

            s = {g.generate_shuffle_X2("d01")};
            min = {g.generate_min("s", "d01")};
            max = {g.generate_max("s", "d01")};
            d01 = {g.generate_blend("min", "max", 0b11001100, 8, asc)};

            s = {g.generate_shuffle_X1("d01")};
            min = {g.generate_min("s", "d01")};
            max = {g.generate_max("s", "d01")};
            d01 = {g.generate_blend("min", "max", 0b10101010, 8, asc)};"""
            self.clean_print(s, file=f)
        print("    }\n", file=f)

    def generate_1v_merge_sorters(self, f: IO, asc: bool):
        g = self
        type = self.type
        maybe_cmp = lambda: ", cmp" if (type == "int64_t" or type == "uint64_t") else ""
        maybe_topbit = lambda: f"\n        TV topBit = _mm256_set1_epi64x(1LLU << 63);" if (
                type == "uint64_t") else ""
        maybe_x1_epi16 = lambda: f"\n        const TV x1 = _mm256_set1_epi64x(0xE'F'C'D'A'B'8'9'6'7'4'5'2'3'0'1);" if (
                type == "uint16_t" or type == "int16_t") else ""

        sfx = "ascending" if asc else "descending"

        s = f"""    static INLINE void merge_01v_{sfx}({g.generate_param_def_list(1)}) {{
            TV min, max, s{maybe_cmp()};{maybe_topbit()}{maybe_x1_epi16()}"""
        print(s, file=f)

        if g.vector_size() == 8:
            s = f"""
            s = {g.generate_shuffle_X4("d01")};
            min = {g.generate_min("s", "d01")};
            max = {g.generate_max("s", "d01")};
            d01 = {g.generate_blend("min", "max", 0b11110000, 8, asc)};"""
            self.clean_print(s, file=f)

        s = f"""
            s = {g.generate_shuffle_X2("d01")};
            {g.crappity_crap_crap("s", "d01")}
            min = {g.generate_min("s", "d01")};
            max = {g.generate_max("s", "d01")};
            d01 = {g.generate_blend("min", "max", 0b1100, 4, asc)};

            s = {g.generate_shuffle_X1("d01")};
            {g.crappity_crap_crap("s", "d01")}
            min = {g.generate_min("s", "d01")};
            max = {g.generate_max("s", "d01")};
            d01 = {g.generate_blend("min", "max", 0b10, 2, asc)};"""

        self.clean_print(s, file=f)
        self.clean_print("    }\n", file=f)

    def generate_compounded_sorter(self, f: IO, width: int, asc: bool, inline: int):
        type = self.type
        g = self
        maybe_cmp = lambda: ", cmp" if (type == "int64_t" or type == "uint64_t") else ""
        maybe_topbit = lambda: f"\n        TV topBit = _mm256_set1_epi64x(1LLU << 63);" if (
                type == "uint64_t") else ""

        w1 = int(next_power_of_2(width) / 2)
        w2 = int(width - w1)

        sfx = "ascending" if asc else "descending"
        rev_sfx = "descending" if asc else "ascending"

        inl = "INLINE" if inline else "NOINLINE"

        s = f"""    static {inl} void sort_{width:02d}v_{sfx}({g.generate_param_def_list(width)}) {{
        TV tmp{maybe_cmp()};{maybe_topbit()}

        sort_{w1:02d}v_{sfx}({g.generate_param_list(1, w1)});
        sort_{w2:02d}v_{rev_sfx}({g.generate_param_list(w1 + 1, w2)});"""

        print(s, file=f)

        for r_n in range(w1 + 1, width + 1):
            l_n = w1 + 1 - (r_n - w1)
            r_var = f"d{r_n:02d}"
            l_var = f"d{l_n:02d}"

            ## We swap the left/right vars according to ascending / descending
            ##if ascending:
            ##    r_var = f"d{r_n:02d}"
            ##    l_var = f"d{l_n:02d}"
            ##else:
            ##    r_var = f"d{l_n:02d}" # This is swapped on purpose!
            ##    l_var = f"d{r_n:02d}" # This is swapped on purpose!

            s = f"""
        tmp = {r_var};
        {g.crappity_crap_crap(f"{l_var}", f"{r_var}")}
        {r_var} = {g.generate_max(f"{l_var}", f"{r_var}")};
        {l_var} = {g.generate_min(f"{l_var}", "tmp")};"""

            self.clean_print(s, file=f)

        s = f"""
        merge_{w1:02d}v_{sfx}({g.generate_param_list(1, w1)});
        merge_{w2:02d}v_{sfx}({g.generate_param_list(w1 + 1, w2)});"""
        self.clean_print(s, file=f)
        self.clean_print("    }\n", file=f)

    def generate_compounded_merger(self, f: IO, width: int, asc: bool, inline: int):
        type = self.type
        g = self
        maybe_cmp = lambda: ", cmp" if (type == "int64_t" or type == "uint64_t") else ""
        maybe_topbit = lambda: f"\n        TV topBit = _mm256_set1_epi64x(1LLU << 63);" if (
                type == "uint64_t") else ""

        w1 = int(next_power_of_2(width) / 2)
        w2 = int(width - w1)

        sfx = "ascending" if asc else "descending"

        inl = "INLINE" if inline else "NOINLINE"

        s = f"""    static {inl} void merge_{width:02d}v_{sfx}({g.generate_param_def_list(width)}) {{
        TV tmp{maybe_cmp()};{maybe_topbit()}"""
        print(s, file=f)

        for r in range(w1 + 1, width + 1):
            x = r - w1
            s = f"""
        tmp = d{x:02d};
        {g.crappity_crap_crap(f"d{r:02d}", f"d{x:02d}")}
        d{x:02d} = {g.generate_min(f"d{r:02d}", f"d{x:02d}")};
        {g.crappity_crap_crap(f"d{r:02d}", "tmp")}
        d{r:02d} = {g.generate_max(f"d{r:02d}", "tmp")};"""
            self.clean_print(s, file=f)

        s = f"""
        merge_{w1:02d}v_{sfx}({g.generate_param_list(1, w1)});
        merge_{w2:02d}v_{sfx}({g.generate_param_list(w1 + 1, w2)});"""
        self.clean_print(s, file=f)
        self.clean_print("    }\n", file=f)

    def generate_cross_min_max(self, f: IO):
        g = self
        type = self.type
        maybe_cmp = lambda: ", cmp" if (type == "int64_t" or type == "uint64_t") else ""
        maybe_topbit = lambda: f"\n        TV topBit = _mm256_set1_epi64x(1LLU << 63);" if (type == "uint64_t") else ""

        s = f"""    static INLINE void cross_min_max(TV& d01, TV& d02) {{
        TV tmp{maybe_cmp()};{maybe_topbit()}

        tmp = {g.generate_reverse("d02")};
        {g.crappity_crap_crap("d01", "tmp")}
        d02 = {g.generate_max("d01", "tmp")};        
        d01 = {g.generate_min("d01", "tmp")};    
    }}\n"""
        self.clean_print(s, file=f)

    def generate_strided_min_max(self, f: IO):
        g = self
        type = self.type
        maybe_cmp = lambda: ", cmp" if (type == "int64_t" or type == "uint64_t") else ""
        maybe_topbit = lambda: f"\n        TV topBit = _mm256_set1_epi64x(1LLU << 63);" if (type == "uint64_t") else ""

        s = f"""    static INLINE void strided_min_max(TV& dl, TV& dr) {{
        TV tmp{maybe_cmp()};{maybe_topbit()}
        
        tmp = dl;
        {g.crappity_crap_crap("dr", "dl")}
        dl = {g.generate_min("dr", "dl")};
        {g.crappity_crap_crap("dr", "tmp")}
        dr = {g.generate_max("dr", "tmp")};
    }}\n"""
        self.clean_print(s, file=f)

    def generate_entry_points_full_vectors(self, f : IO, asc: bool):
        type = self.type
        g = self
        sfx = "ascending" if asc else "descending"
        for m in range(1, g.max_bitonic_sort_vectors() + 1):
            s = f"""
    // This is generated for testing purposes only
    static NOINLINE void sort_{m:02d}v_full_{sfx}({type} *ptr) {{"""
            self.clean_print(s, file=f)

            for l in range(0, m):
                s = f"        TV d{l + 1:02d} = {g.get_load_intrinsic('ptr', l)};"
                print(s, file=f)

            s = f"        sort_{m:02d}v_{sfx}({g.generate_param_list(1, m)});"
            self.clean_print(s, file=f)

            for l in range(0, m):
                s = f"        {g.get_store_intrinsic('ptr', l, f'd{l + 1:02d}')};"
                self.clean_print(s, file=f)

            self.clean_print("    }\n", file=f)

    def generate_entry_points_partial(self, f: IO):
        type = self.type
        g = self
        for m in range(1, g.max_bitonic_sort_vectors() + 1):
            s = f"""
    static NOINLINE void sort_{m:02d}v_partial({type} *ptr, int remainder) {{
        assert(remainder * N < sizeof(mask_table_{self.vector_size()}));
        const auto mask = _mm256_cvtepi8_epi{int(256 / self.vector_size())}(_mm_loadu_si128((__m128i*)(mask_table_{self.vector_size()} + remainder * N)));
"""
            self.clean_print(s, file=f)

            for l in range(0, m-1):
                s = f"        TV d{l + 1:02d} = {g.get_load_intrinsic('ptr', l)};"
                print(s, file=f)
            s = f"        TV d{m:02d} = {g.get_mask_load_intrinsic('ptr', m - 1, 'mask')};"
            self.clean_print(s, file=f)

            s = f"        sort_{m:02d}v_ascending({g.generate_param_list(1, m)});"
            self.clean_print(s, file=f)

            for l in range(0, m-1):
                s = f"        {g.get_store_intrinsic('ptr', l, f'd{l + 1:02d}')};"
                print(s, file=f)
            s = f"        {g.get_mask_store_intrinsic('ptr', m - 1, f'd{m:02d}', 'mask')};"
            self.clean_print(s, file=f)

            self.clean_print("    }", file=f)

    def generate_master_entry_point_full(self, f: IO, asc: bool):
        basename = os.path.basename(f.name)
        t = self.type

        sfx = "ascending" if asc else "descending"

        s = f"""
    // This is generated for testing purposes only
    static void sort_full_vectors_{sfx}({t} *ptr, size_t length) {{
        assert(length % N == 0);
        switch(length / N) {{"""
        self.clean_print(s, file=f)

        for m in range(1, self.max_bitonic_sort_vectors() + 1):
            s = f"            case {m}: sort_{m:02d}v_full_{sfx}(ptr); break;"
            print(s, file=f)
        self.clean_print("        }", file=f)
        self.clean_print("    }\n", file=f)

    #     s = f"""void vxsort::smallsort::bitonic<{t}, vector_machine::AVX2 >::sort({t} *ptr, size_t length) {{
    # const auto fullvlength = length / N;
    # const int remainder = (int) (length - fullvlength * N);
    # const auto v = fullvlength + ((remainder > 0) ? 1 : 0);
    # switch(v) {{"""
    #     print(s, file=f_src)
    #
    #     for m in range(1, self.max_bitonic_sort_vectors() + 1):
    #         s = f"        case {m}: sort_{m:02d}v_alt(ptr, remainder); break;"
    #         print(s, file=f_src)
    #     print("    }", file=f_src)
    #     print("}", file=f_src)