/////////////////////////////////////////////////////////////////////////////
////
// This file was auto-generated by a tool at 2020-08-17 20:51:06
//
// It is recommended you DO NOT directly edit this file but instead edit
// the code-generator that generated this source file instead.
/////////////////////////////////////////////////////////////////////////////

#ifndef BITONIC_MACHINE_AVX512_INT64_T_H
#define BITONIC_MACHINE_AVX512_INT64_T_H


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
template<> struct bitonic_machine<int64_t, AVX512> {
    static const int N = 8;
    static constexpr int64_t MAX = std::numeric_limits<int64_t>::max();
public:
    typedef __m512i TV;
    typedef __mmask8 TMASK;


    static INLINE void sort_01v_ascending(TV& d01) {
        TV  min, s;

        s = d2i(_mm512_permute_pd(i2d(d01), _MM_PERM_BBBB));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x00AA, s, d01);

        s = d2i(_mm512_permutex_pd(i2d(d01), _MM_PERM_ABCD));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x00CC, s, d01);

        s = d2i(_mm512_permute_pd(i2d(d01), _MM_PERM_BBBB));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x00AA, s, d01);

        s = d2i(_mm512_shuffle_f64x2(_mm512_permutex_pd(i2d(d01), _MM_PERM_ABCD), _mm512_permutex_pd(i2d(d01), _MM_PERM_ABCD), _MM_PERM_BADC));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x00F0, s, d01);

        s = d2i(_mm512_permutex_pd(i2d(d01), _MM_PERM_BADC));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x00CC, s, d01);

        s = d2i(_mm512_permute_pd(i2d(d01), _MM_PERM_BBBB));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x00AA, s, d01);
    }
    static INLINE void merge_01v_ascending(TV& d01) {
        TV  min, s;

        s = d2i(_mm512_shuffle_f64x2(i2d(d01), i2d(d01), _MM_PERM_BADC));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x00F0, s, d01);

        s = d2i(_mm512_permutex_pd(i2d(d01), _MM_PERM_BADC));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x00CC, s, d01);

        s = d2i(_mm512_permute_pd(i2d(d01), _MM_PERM_BBBB));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x00AA, s, d01);
    }
    static INLINE void sort_01v_descending(TV& d01) {
        TV  min, s;

        s = d2i(_mm512_permute_pd(i2d(d01), _MM_PERM_BBBB));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x0055, s, d01);

        s = d2i(_mm512_permutex_pd(i2d(d01), _MM_PERM_ABCD));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x0033, s, d01);

        s = d2i(_mm512_permute_pd(i2d(d01), _MM_PERM_BBBB));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x0055, s, d01);

        s = d2i(_mm512_shuffle_f64x2(_mm512_permutex_pd(i2d(d01), _MM_PERM_ABCD), _mm512_permutex_pd(i2d(d01), _MM_PERM_ABCD), _MM_PERM_BADC));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x000F, s, d01);

        s = d2i(_mm512_permutex_pd(i2d(d01), _MM_PERM_BADC));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x0033, s, d01);

        s = d2i(_mm512_permute_pd(i2d(d01), _MM_PERM_BBBB));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x0055, s, d01);
    }
    static INLINE void merge_01v_descending(TV& d01) {
        TV  min, s;

        s = d2i(_mm512_shuffle_f64x2(i2d(d01), i2d(d01), _MM_PERM_BADC));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x000F, s, d01);

        s = d2i(_mm512_permutex_pd(i2d(d01), _MM_PERM_BADC));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x0033, s, d01);

        s = d2i(_mm512_permute_pd(i2d(d01), _MM_PERM_BBBB));
        min = _mm512_min_epi64(s, d01);
        d01 = _mm512_mask_max_epi64(min, 0x0055, s, d01);
    }
    static INLINE void sort_02v_ascending(TV& d01, TV& d02) {
        TV tmp;

        sort_01v_ascending(d01);
        sort_01v_descending(d02);

        tmp = d02;
        d02 = _mm512_max_epi64(d01, d02);
        d01 = _mm512_min_epi64(d01, tmp);

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void sort_02v_descending(TV& d01, TV& d02) {
        TV tmp;

        sort_01v_descending(d01);
        sort_01v_ascending(d02);

        tmp = d02;
        d02 = _mm512_max_epi64(d01, d02);
        d01 = _mm512_min_epi64(d01, tmp);

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void merge_02v_ascending(TV& d01, TV& d02) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi64(d02, d01);
        d02 = _mm512_max_epi64(d02, tmp);

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void merge_02v_descending(TV& d01, TV& d02) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi64(d02, d01);
        d02 = _mm512_max_epi64(d02, tmp);

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void sort_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        sort_02v_ascending(d01, d02);
        sort_01v_descending(d03);

        tmp = d03;
        d03 = _mm512_max_epi64(d02, d03);
        d02 = _mm512_min_epi64(d02, tmp);

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void sort_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        sort_02v_descending(d01, d02);
        sort_01v_ascending(d03);

        tmp = d03;
        d03 = _mm512_max_epi64(d02, d03);
        d02 = _mm512_min_epi64(d02, tmp);

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void merge_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi64(d03, d01);
        d03 = _mm512_max_epi64(d03, tmp);

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void merge_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi64(d03, d01);
        d03 = _mm512_max_epi64(d03, tmp);

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void sort_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        sort_02v_ascending(d01, d02);
        sort_02v_descending(d03, d04);

        tmp = d03;
        d03 = _mm512_max_epi64(d02, d03);
        d02 = _mm512_min_epi64(d02, tmp);

        tmp = d04;
        d04 = _mm512_max_epi64(d01, d04);
        d01 = _mm512_min_epi64(d01, tmp);

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void sort_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        sort_02v_descending(d01, d02);
        sort_02v_ascending(d03, d04);

        tmp = d03;
        d03 = _mm512_max_epi64(d02, d03);
        d02 = _mm512_min_epi64(d02, tmp);

        tmp = d04;
        d04 = _mm512_max_epi64(d01, d04);
        d01 = _mm512_min_epi64(d01, tmp);

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void merge_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi64(d03, d01);
        d03 = _mm512_max_epi64(d03, tmp);

        tmp = d02;
        d02 = _mm512_min_epi64(d04, d02);
        d04 = _mm512_max_epi64(d04, tmp);

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void merge_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        tmp = d01;
        d01 = _mm512_min_epi64(d03, d01);
        d03 = _mm512_max_epi64(d03, tmp);

        tmp = d02;
        d02 = _mm512_min_epi64(d04, d02);
        d04 = _mm512_max_epi64(d04, tmp);

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void cross_min_max(TV& d01, TV& d02) {
        TV tmp;

        tmp = d2i(_mm512_shuffle_pd(i2d(d02), i2d(d02), 0xB1));
        d02 = _mm512_max_epi64(d01, tmp);
        d01 = _mm512_min_epi64(d01, tmp);
        }
    static INLINE void strided_min_max(TV& d01, TV& d02) {
        TV tmp;
        
        tmp = d01;
        d01 = _mm512_min_epi64(d02, d01);
        d02 = _mm512_max_epi64(d02, tmp);
    }

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

