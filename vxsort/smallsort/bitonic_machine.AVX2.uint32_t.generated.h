/////////////////////////////////////////////////////////////////////////////
////
// This file was auto-generated by a tool at 2020-08-17 20:51:06
//
// It is recommended you DO NOT directly edit this file but instead edit
// the code-generator that generated this source file instead.
/////////////////////////////////////////////////////////////////////////////

#ifndef BITONIC_MACHINE_AVX2_UINT32_T_H
#define BITONIC_MACHINE_AVX2_UINT32_T_H

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

namespace vxsort {
namespace smallsort {

template<> struct bitonic_machine<uint32_t, AVX2> {
    static const int N = 8;
    static constexpr uint32_t MAX = std::numeric_limits<uint32_t>::max();
public:
    typedef __m256i TV;
    typedef __m256i TMASK;


    static INLINE void sort_01v_ascending(TV& d01) {
            TV min, max, s;

            s = _mm256_shuffle_epi32(d01, 0xB1);
            
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(min, max, 0xAA);

            s = _mm256_shuffle_epi32(d01, 0x1B);
            
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(min, max, 0xCC);

            s = _mm256_shuffle_epi32(d01, 0xB1);
            
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(min, max, 0xAA);

            s = d2i(_mm256_permute4x64_pd(i2d(_mm256_shuffle_epi32(d01, 0x1B)), 0x4E));
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(min, max, 0xF0);

            s = _mm256_shuffle_epi32(d01, 0x4E);
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(min, max, 0xCC);

            s = _mm256_shuffle_epi32(d01, 0xB1);
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(min, max, 0xAA);
}
    static INLINE void merge_01v_ascending(TV& d01) {
            TV min, max, s;

            s = d2i(_mm256_permute4x64_pd(i2d(d01), 0x4E));
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(min, max, 0xF0);

            s = _mm256_shuffle_epi32(d01, 0x4E);
            
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(min, max, 0xCC);

            s = _mm256_shuffle_epi32(d01, 0xB1);
            
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(min, max, 0xAA);
    }
    static INLINE void sort_01v_descending(TV& d01) {
            TV min, max, s;

            s = _mm256_shuffle_epi32(d01, 0xB1);
            
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(max, min, 0xAA);

            s = _mm256_shuffle_epi32(d01, 0x1B);
            
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(max, min, 0xCC);

            s = _mm256_shuffle_epi32(d01, 0xB1);
            
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(max, min, 0xAA);

            s = d2i(_mm256_permute4x64_pd(i2d(_mm256_shuffle_epi32(d01, 0x1B)), 0x4E));
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(max, min, 0xF0);

            s = _mm256_shuffle_epi32(d01, 0x4E);
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(max, min, 0xCC);

            s = _mm256_shuffle_epi32(d01, 0xB1);
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(max, min, 0xAA);
}
    static INLINE void merge_01v_descending(TV& d01) {
            TV min, max, s;

            s = d2i(_mm256_permute4x64_pd(i2d(d01), 0x4E));
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(max, min, 0xF0);

            s = _mm256_shuffle_epi32(d01, 0x4E);
            
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(max, min, 0xCC);

            s = _mm256_shuffle_epi32(d01, 0xB1);
            
            min = _mm256_min_epu32(s, d01);
            max = _mm256_max_epu32(s, d01);
            d01 = _mm256_blend_epi32(max, min, 0xAA);
    }
    static INLINE void sort_02v_ascending(TV& d01, TV& d02) {
        TV tmp;

        sort_01v_ascending(d01);
        sort_01v_descending(d02);

        tmp = d02;
        
        d02 = _mm256_max_epu32(d01, d02);
        d01 = _mm256_min_epu32(d01, tmp);

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void sort_02v_descending(TV& d01, TV& d02) {
        TV tmp;

        sort_01v_descending(d01);
        sort_01v_ascending(d02);

        tmp = d02;
        
        d02 = _mm256_max_epu32(d01, d02);
        d01 = _mm256_min_epu32(d01, tmp);

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void merge_02v_ascending(TV& d01, TV& d02) {
        TV tmp;

        tmp = d01;
        
        d01 = _mm256_min_epu32(d02, d01);
        
        d02 = _mm256_max_epu32(d02, tmp);

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void merge_02v_descending(TV& d01, TV& d02) {
        TV tmp;

        tmp = d01;
        
        d01 = _mm256_min_epu32(d02, d01);
        
        d02 = _mm256_max_epu32(d02, tmp);

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void sort_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        sort_02v_ascending(d01, d02);
        sort_01v_descending(d03);

        tmp = d03;
        
        d03 = _mm256_max_epu32(d02, d03);
        d02 = _mm256_min_epu32(d02, tmp);

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void sort_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        sort_02v_descending(d01, d02);
        sort_01v_ascending(d03);

        tmp = d03;
        
        d03 = _mm256_max_epu32(d02, d03);
        d02 = _mm256_min_epu32(d02, tmp);

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void merge_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        tmp = d01;
        
        d01 = _mm256_min_epu32(d03, d01);
        
        d03 = _mm256_max_epu32(d03, tmp);

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void merge_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp;

        tmp = d01;
        
        d01 = _mm256_min_epu32(d03, d01);
        
        d03 = _mm256_max_epu32(d03, tmp);

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void sort_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        sort_02v_ascending(d01, d02);
        sort_02v_descending(d03, d04);

        tmp = d03;
        
        d03 = _mm256_max_epu32(d02, d03);
        d02 = _mm256_min_epu32(d02, tmp);

        tmp = d04;
        
        d04 = _mm256_max_epu32(d01, d04);
        d01 = _mm256_min_epu32(d01, tmp);

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void sort_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        sort_02v_descending(d01, d02);
        sort_02v_ascending(d03, d04);

        tmp = d03;
        
        d03 = _mm256_max_epu32(d02, d03);
        d02 = _mm256_min_epu32(d02, tmp);

        tmp = d04;
        
        d04 = _mm256_max_epu32(d01, d04);
        d01 = _mm256_min_epu32(d01, tmp);

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void merge_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        tmp = d01;
        
        d01 = _mm256_min_epu32(d03, d01);
        
        d03 = _mm256_max_epu32(d03, tmp);

        tmp = d02;
        
        d02 = _mm256_min_epu32(d04, d02);
        
        d04 = _mm256_max_epu32(d04, tmp);

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void merge_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp;

        tmp = d01;
        
        d01 = _mm256_min_epu32(d03, d01);
        
        d03 = _mm256_max_epu32(d03, tmp);

        tmp = d02;
        
        d02 = _mm256_min_epu32(d04, d02);
        
        d04 = _mm256_max_epu32(d04, tmp);

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void cross_min_max(TV& d01, TV& d02) {
        TV tmp;

        tmp = d2i(_mm256_permute4x64_pd(i2d(_mm256_shuffle_epi32(d02, 0x1B)), 0x4E));
        
        d02 = _mm256_max_epu32(d01, tmp);        
        d01 = _mm256_min_epu32(d01, tmp);    
    }
    static INLINE void strided_min_max(TV& dl, TV& dr) {
        TV tmp;
        
        tmp = dl;
        
        dl = _mm256_min_epu32(dr, dl);
        
        dr = _mm256_max_epu32(dr, tmp);
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
    
