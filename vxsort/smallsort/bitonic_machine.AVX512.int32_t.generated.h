/////////////////////////////////////////////////////////////////////////////
////
// This file was auto-generated by a tool at 2021-01-02 20:19:37
//
// It is recommended you DO NOT directly edit this file but instead edit
// the code-generator that generated this source file instead.
/////////////////////////////////////////////////////////////////////////////

#ifndef BITONIC_MACHINE_AVX512_INT32_T_H
#define BITONIC_MACHINE_AVX512_INT32_T_H


#ifdef __GNUC__
#ifdef __clang__
#pragma clang attribute push (__attribute__((target("avx512f"))), apply_to = any(function))
#else
#pragma GCC push_options
#pragma GCC target("avx512f")
#endif
#endif

#include <limits>
#include <immintrin.h>
#include "bitonic_machine.h"

#define i2d _mm512_castsi512_pd
#define d2i _mm512_castpd_si512
#define i2s _mm512_castsi512_ps
#define s2i _mm512_castps_si512
#define s2d _mm512_castps_pd
#define d2s _mm521_castpd_ps

namespace vxsort {
namespace smallsort {
template<> struct bitonic_machine<int32_t, AVX512> {
    static const int N = 16;
    static constexpr int32_t MAX = std::numeric_limits<int32_t>::max();
public:
    typedef __m512i TV;
    typedef __mmask16 TMASK;

    static INLINE void sort_01v_ascending(TV& d01) {
        TV  min, s;

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0110011001100110, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0011110000111100, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0101101001011010, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0000111111110000, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0011001111001100, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0101010110101010, s, d01);

        s = _mm512_shuffle_i64x2(d01, d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1111111100000000, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1111000011110000, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1100110011001100, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1010101010101010, s, d01);
    }
    static INLINE void merge_01v_ascending(TV& d01) {
        TV  min, s;

        s = _mm512_shuffle_i64x2(d01, d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1111111100000000, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1111000011110000, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1100110011001100, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1010101010101010, s, d01);
    }
    static INLINE void sort_01v_descending(TV& d01) {
        TV  min, s;

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1001100110011001, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1100001111000011, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1010010110100101, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1111000000001111, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1100110000110011, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b1010101001010101, s, d01);

        s = _mm512_shuffle_i64x2(d01, d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0000000011111111, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0000111100001111, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0011001100110011, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0101010101010101, s, d01);
    }
    static INLINE void merge_01v_descending(TV& d01) {
        TV  min, s;

        s = _mm512_shuffle_i64x2(d01, d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0000000011111111, s, d01);

        s = _mm512_permutex_epi64(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0000111100001111, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM) 0b01'00'11'10);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0011001100110011, s, d01);

        s = _mm512_shuffle_epi32(d01, (_MM_PERM_ENUM)  0b10'11'00'01);
        min = _mm512_min_epi32(s, d01);
        d01 = _mm512_mask_max_epi32(min, 0b0101010101010101, s, d01);
    }
    static INLINE void sort_02v_ascending(TV& d01, TV& d02) {
        TV tmp;

        sort_01v_ascending(d01);
        sort_01v_descending(d02);

        tmp = d02;
        d02 = _mm512_max_epi32(d01, d02);
        d01 = _mm512_min_epi32(d01, tmp);

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void sort_02v_descending(TV& d01, TV& d02) {
        TV tmp;

        sort_01v_descending(d01);
        sort_01v_ascending(d02);

        tmp = d02;
        d02 = _mm512_max_epi32(d01, d02);
        d01 = _mm512_min_epi32(d01, tmp);

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void merge_02v_ascending(TV& d01, TV& d02) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi32(d02, d01);
        d02 = _mm512_max_epi32(d02, tmp);

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void merge_02v_descending(TV& d01, TV& d02) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi32(d02, d01);
        d02 = _mm512_max_epi32(d02, tmp);

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void sort_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        sort_02v_ascending(d01, d02);
        sort_01v_descending(d03);

        tmp = d03;
        d03 = _mm512_max_epi32(d02, d03);
        d02 = _mm512_min_epi32(d02, tmp);

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void sort_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        sort_02v_descending(d01, d02);
        sort_01v_ascending(d03);

        tmp = d03;
        d03 = _mm512_max_epi32(d02, d03);
        d02 = _mm512_min_epi32(d02, tmp);

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void merge_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi32(d03, d01);
        d03 = _mm512_max_epi32(d03, tmp);

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void merge_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi32(d03, d01);
        d03 = _mm512_max_epi32(d03, tmp);

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void sort_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        sort_02v_ascending(d01, d02);
        sort_02v_descending(d03, d04);

        tmp = d03;
        d03 = _mm512_max_epi32(d02, d03);
        d02 = _mm512_min_epi32(d02, tmp);

        tmp = d04;
        d04 = _mm512_max_epi32(d01, d04);
        d01 = _mm512_min_epi32(d01, tmp);

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void sort_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        sort_02v_descending(d01, d02);
        sort_02v_ascending(d03, d04);

        tmp = d03;
        d03 = _mm512_max_epi32(d02, d03);
        d02 = _mm512_min_epi32(d02, tmp);

        tmp = d04;
        d04 = _mm512_max_epi32(d01, d04);
        d01 = _mm512_min_epi32(d01, tmp);

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void merge_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi32(d03, d01);
        d03 = _mm512_max_epi32(d03, tmp);

        tmp = d02;
        d02 = _mm512_min_epi32(d04, d02);
        d04 = _mm512_max_epi32(d04, tmp);

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void merge_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi32(d03, d01);
        d03 = _mm512_max_epi32(d03, tmp);

        tmp = d02;
        d02 = _mm512_min_epi32(d04, d02);
        d04 = _mm512_max_epi32(d04, tmp);

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void cross_min_max(TV& d01, TV& d02) {
        TV tmp;

        tmp = _mm512_shuffle_i64x2(_mm512_shuffle_epi32(d02, (_MM_PERM_ENUM) 0b00'01'10'11), _mm512_shuffle_epi32(d02, (_MM_PERM_ENUM) 0b00'01'10'11), (_MM_PERM_ENUM) 0b00'01'10'11);
        d02 = _mm512_max_epi32(d01, tmp);
        d01 = _mm512_min_epi32(d01, tmp);
        }
    static INLINE void strided_min_max(TV& d01, TV& d02) {
        TV tmp;
        
        tmp = d01;
        d01 = _mm512_min_epi32(d02, d01);
        d02 = _mm512_max_epi32(d02, tmp);
    }

#ifdef BITONIC_TESTS

    // This is generated for testing purposes only
    static NOINLINE void sort_01v_full_ascending(int32_t *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        sort_01v_ascending(d01);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_02v_full_ascending(int32_t *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        sort_02v_ascending(d01, d02);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_03v_full_ascending(int32_t *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        TV d03 = _mm512_loadu_si512((__m512i const *) ptr + 2);;
        sort_03v_ascending(d01, d02, d03);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
        _mm512_storeu_si512((__m512i *) ptr + 2, d03);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_04v_full_ascending(int32_t *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        TV d03 = _mm512_loadu_si512((__m512i const *) ptr + 2);;
        TV d04 = _mm512_loadu_si512((__m512i const *) ptr + 3);;
        sort_04v_ascending(d01, d02, d03, d04);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
        _mm512_storeu_si512((__m512i *) ptr + 2, d03);
        _mm512_storeu_si512((__m512i *) ptr + 3, d04);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_01v_full_descending(int32_t *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        sort_01v_descending(d01);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_02v_full_descending(int32_t *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        sort_02v_descending(d01, d02);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_03v_full_descending(int32_t *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        TV d03 = _mm512_loadu_si512((__m512i const *) ptr + 2);;
        sort_03v_descending(d01, d02, d03);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
        _mm512_storeu_si512((__m512i *) ptr + 2, d03);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_04v_full_descending(int32_t *ptr) {
        TV d01 = _mm512_loadu_si512((__m512i const *) ptr + 0);;
        TV d02 = _mm512_loadu_si512((__m512i const *) ptr + 1);;
        TV d03 = _mm512_loadu_si512((__m512i const *) ptr + 2);;
        TV d04 = _mm512_loadu_si512((__m512i const *) ptr + 3);;
        sort_04v_descending(d01, d02, d03, d04);
        _mm512_storeu_si512((__m512i *) ptr + 0, d01);
        _mm512_storeu_si512((__m512i *) ptr + 1, d02);
        _mm512_storeu_si512((__m512i *) ptr + 2, d03);
        _mm512_storeu_si512((__m512i *) ptr + 3, d04);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_full_vectors_ascending(int32_t *ptr, size_t length) {
        assert(length % N == 0);
        switch(length / N) {
            case 1: sort_01v_full_ascending(ptr); break;
            case 2: sort_02v_full_ascending(ptr); break;
            case 3: sort_03v_full_ascending(ptr); break;
            case 4: sort_04v_full_ascending(ptr); break;
        }
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_full_vectors_descending(int32_t *ptr, size_t length) {
        assert(length % N == 0);
        switch(length / N) {
            case 1: sort_01v_full_descending(ptr); break;
            case 2: sort_02v_full_descending(ptr); break;
            case 3: sort_03v_full_descending(ptr); break;
            case 4: sort_04v_full_descending(ptr); break;
        }
    }

#endif

};
}
}

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
