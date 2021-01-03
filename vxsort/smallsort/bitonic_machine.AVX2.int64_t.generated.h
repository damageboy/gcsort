/////////////////////////////////////////////////////////////////////////////
////
// This file was auto-generated by a tool at 2021-01-02 20:19:37
//
// It is recommended you DO NOT directly edit this file but instead edit
// the code-generator that generated this source file instead.
/////////////////////////////////////////////////////////////////////////////

#ifndef BITONIC_MACHINE_AVX2_INT64_T_H
#define BITONIC_MACHINE_AVX2_INT64_T_H

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

template<> struct bitonic_machine<int64_t, AVX2> {
    static const int N = 4;
    static constexpr int64_t MAX = std::numeric_limits<int64_t>::max();
public:
    typedef __m256i TV;
    typedef __m256i TMASK;

    static INLINE void sort_01v_ascending(TV& d01) {
        TV min, max, s;
        TV cmp;

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(s, d01);
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00000110));

        s = d2i(_mm256_permute4x64_pd(i2d(d01), 0b01'00'11'10));
        cmp = _mm256_cmpgt_epi64(s, d01);
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00001100));

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(s, d01);
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00001010));
    }
    static INLINE void merge_01v_ascending(TV& d01) {
        TV min, max, s;
        TV cmp;

        s = d2i(_mm256_permute4x64_pd(i2d(d01), 0b01'00'11'10));
        cmp = _mm256_cmpgt_epi64(s, d01);
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00001100));

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(s, d01);
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00001010));
    }
    static INLINE void sort_01v_descending(TV& d01) {
        TV min, max, s;
        TV cmp;

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(s, d01);
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00001001));

        s = d2i(_mm256_permute4x64_pd(i2d(d01), 0b01'00'11'10));
        cmp = _mm256_cmpgt_epi64(s, d01);
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00000011));

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(s, d01);
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00000101));
    }
    static INLINE void merge_01v_descending(TV& d01) {
        TV min, max, s;
        TV cmp;

        s = d2i(_mm256_permute4x64_pd(i2d(d01), 0b01'00'11'10));
        cmp = _mm256_cmpgt_epi64(s, d01);
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00000011));

        s = d2i(_mm256_shuffle_pd(i2d(d01), i2d(d01), 0b0'1'0'1));
        cmp = _mm256_cmpgt_epi64(s, d01);
        min = d2i(_mm256_blendv_pd(i2d(s), i2d(d01), i2d(cmp)));
        max = d2i(_mm256_blendv_pd(i2d(d01), i2d(s), i2d(cmp)));
        d01 = d2i(_mm256_blend_pd(i2d(min), i2d(max), 0b00000101));
    }
    static INLINE void sort_02v_ascending(TV& d01, TV& d02) {
        TV tmp, cmp;

        sort_01v_ascending(d01);
        sort_01v_descending(d02);

        tmp = d02;
        cmp = _mm256_cmpgt_epi64(d01, d02);
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(d01), i2d(cmp)));
        d01 = d2i(_mm256_blendv_pd(i2d(d01), i2d(tmp), i2d(cmp)));

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void sort_02v_descending(TV& d01, TV& d02) {
        TV tmp, cmp;

        sort_01v_descending(d01);
        sort_01v_ascending(d02);

        tmp = d02;
        cmp = _mm256_cmpgt_epi64(d01, d02);
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(d01), i2d(cmp)));
        d01 = d2i(_mm256_blendv_pd(i2d(d01), i2d(tmp), i2d(cmp)));

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void merge_02v_ascending(TV& d01, TV& d02) {
        TV tmp, cmp;

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(d02, d01);
        d01 = d2i(_mm256_blendv_pd(i2d(d02), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(d02, tmp);
        d02 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d02), i2d(cmp)));

        merge_01v_ascending(d01);
        merge_01v_ascending(d02);
    }
    static INLINE void merge_02v_descending(TV& d01, TV& d02) {
        TV tmp, cmp;

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(d02, d01);
        d01 = d2i(_mm256_blendv_pd(i2d(d02), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(d02, tmp);
        d02 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d02), i2d(cmp)));

        merge_01v_descending(d01);
        merge_01v_descending(d02);
    }
    static INLINE void sort_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp, cmp;

        sort_02v_ascending(d01, d02);
        sort_01v_descending(d03);

        tmp = d03;
        cmp = _mm256_cmpgt_epi64(d02, d03);
        d03 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d02), i2d(cmp)));
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(tmp), i2d(cmp)));

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void sort_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp, cmp;

        sort_02v_descending(d01, d02);
        sort_01v_ascending(d03);

        tmp = d03;
        cmp = _mm256_cmpgt_epi64(d02, d03);
        d03 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d02), i2d(cmp)));
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(tmp), i2d(cmp)));

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void merge_03v_ascending(TV& d01, TV& d02, TV& d03) {
        TV tmp, cmp;

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(d03, d01);
        d01 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(d03, tmp);
        d03 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d03), i2d(cmp)));

        merge_02v_ascending(d01, d02);
        merge_01v_ascending(d03);
    }
    static INLINE void merge_03v_descending(TV& d01, TV& d02, TV& d03) {
        TV tmp, cmp;

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(d03, d01);
        d01 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(d03, tmp);
        d03 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d03), i2d(cmp)));

        merge_02v_descending(d01, d02);
        merge_01v_descending(d03);
    }
    static INLINE void sort_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp, cmp;

        sort_02v_ascending(d01, d02);
        sort_02v_descending(d03, d04);

        tmp = d03;
        cmp = _mm256_cmpgt_epi64(d02, d03);
        d03 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d02), i2d(cmp)));
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(tmp), i2d(cmp)));

        tmp = d04;
        cmp = _mm256_cmpgt_epi64(d01, d04);
        d04 = d2i(_mm256_blendv_pd(i2d(d04), i2d(d01), i2d(cmp)));
        d01 = d2i(_mm256_blendv_pd(i2d(d01), i2d(tmp), i2d(cmp)));

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void sort_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp, cmp;

        sort_02v_descending(d01, d02);
        sort_02v_ascending(d03, d04);

        tmp = d03;
        cmp = _mm256_cmpgt_epi64(d02, d03);
        d03 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d02), i2d(cmp)));
        d02 = d2i(_mm256_blendv_pd(i2d(d02), i2d(tmp), i2d(cmp)));

        tmp = d04;
        cmp = _mm256_cmpgt_epi64(d01, d04);
        d04 = d2i(_mm256_blendv_pd(i2d(d04), i2d(d01), i2d(cmp)));
        d01 = d2i(_mm256_blendv_pd(i2d(d01), i2d(tmp), i2d(cmp)));

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void merge_04v_ascending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp, cmp;

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(d03, d01);
        d01 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(d03, tmp);
        d03 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d03), i2d(cmp)));

        tmp = d02;
        cmp = _mm256_cmpgt_epi64(d04, d02);
        d02 = d2i(_mm256_blendv_pd(i2d(d04), i2d(d02), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(d04, tmp);
        d04 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d04), i2d(cmp)));

        merge_02v_ascending(d01, d02);
        merge_02v_ascending(d03, d04);
    }
    static INLINE void merge_04v_descending(TV& d01, TV& d02, TV& d03, TV& d04) {
        TV tmp, cmp;

        tmp = d01;
        cmp = _mm256_cmpgt_epi64(d03, d01);
        d01 = d2i(_mm256_blendv_pd(i2d(d03), i2d(d01), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(d03, tmp);
        d03 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d03), i2d(cmp)));

        tmp = d02;
        cmp = _mm256_cmpgt_epi64(d04, d02);
        d02 = d2i(_mm256_blendv_pd(i2d(d04), i2d(d02), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(d04, tmp);
        d04 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d04), i2d(cmp)));

        merge_02v_descending(d01, d02);
        merge_02v_descending(d03, d04);
    }
    static INLINE void cross_min_max(TV& d01, TV& d02) {
        TV tmp, cmp;

        tmp = d2i(_mm256_permute4x64_pd(i2d(d02), 0b00'01'10'11));
        cmp = _mm256_cmpgt_epi64(d01, tmp);
        d02 = d2i(_mm256_blendv_pd(i2d(tmp), i2d(d01), i2d(cmp)));        
        d01 = d2i(_mm256_blendv_pd(i2d(d01), i2d(tmp), i2d(cmp)));    
    }
    static INLINE void strided_min_max(TV& dl, TV& dr) {
        TV tmp, cmp;
        
        tmp = dl;
        cmp = _mm256_cmpgt_epi64(dr, dl);
        dl = d2i(_mm256_blendv_pd(i2d(dr), i2d(dl), i2d(cmp)));
        cmp = _mm256_cmpgt_epi64(dr, tmp);
        dr = d2i(_mm256_blendv_pd(i2d(tmp), i2d(dr), i2d(cmp)));
    }

#ifdef BITONIC_TESTS

    // This is generated for testing purposes only
    static NOINLINE void sort_01v_full_ascending(int64_t *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        sort_01v_ascending(d01);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_02v_full_ascending(int64_t *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        sort_02v_ascending(d01, d02);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_03v_full_ascending(int64_t *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        TV d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);;
        sort_03v_ascending(d01, d02, d03);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
        _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_04v_full_ascending(int64_t *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        TV d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);;
        TV d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);;
        sort_04v_ascending(d01, d02, d03, d04);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
        _mm256_storeu_si256((__m256i *) ptr + 2, d03);
        _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_01v_full_descending(int64_t *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        sort_01v_descending(d01);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_02v_full_descending(int64_t *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        sort_02v_descending(d01, d02);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_03v_full_descending(int64_t *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        TV d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);;
        sort_03v_descending(d01, d02, d03);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
        _mm256_storeu_si256((__m256i *) ptr + 2, d03);
    }

    // This is generated for testing purposes only
    static NOINLINE void sort_04v_full_descending(int64_t *ptr) {
        TV d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);;
        TV d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);;
        TV d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);;
        TV d04 = _mm256_lddqu_si256((__m256i const *) ptr + 3);;
        sort_04v_descending(d01, d02, d03, d04);
        _mm256_storeu_si256((__m256i *) ptr + 0, d01);
        _mm256_storeu_si256((__m256i *) ptr + 1, d02);
        _mm256_storeu_si256((__m256i *) ptr + 2, d03);
        _mm256_storeu_si256((__m256i *) ptr + 3, d04);
    }

    // This is generated for testing purposes only
    static void sort_full_vectors_ascending(int64_t *ptr, size_t length) {
        assert(length % N == 0);
        switch(length / N) {
            case 1: sort_01v_full_ascending(ptr); break;
            case 2: sort_02v_full_ascending(ptr); break;
            case 3: sort_03v_full_ascending(ptr); break;
            case 4: sort_04v_full_ascending(ptr); break;
        }
    }

    // This is generated for testing purposes only
    static void sort_full_vectors_descending(int64_t *ptr, size_t length) {
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
