
#include <immintrin.h>

namespace gcsort {
namespace smallsort {


static inline void bitonic_sort_float_01v_ascending(__m256& d01) {
    __m256  min, max, s;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_shuffle_epi32(d01, 0xB1);
    
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(min, max, 0xAA);

    s = _mm256_shuffle_epi32(d01, 0x1B);
    
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(min, max, 0xCC);

    s = _mm256_shuffle_epi32(d01, 0xB1);
    
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(min, max, 0xAA);

    s = _mm256_permute4x64_pd((__m256d) _mm256_shuffle_epi32(d01, 0x1B), 0x4E);
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(min, max, 0xF0);

    s = _mm256_shuffle_epi32(d01, 0x4E);
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(min, max, 0xCC);

    s = _mm256_shuffle_epi32(d01, 0xB1);
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(min, max, 0xAA);
}

static inline void bitonic_sort_float_01v_merge_ascending(__m256& d01) {
    __m256  min, max, s;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_permute4x64_pd((__m256d) d01, 0x4E);
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(min, max, 0xF0);

    s = _mm256_shuffle_epi32(d01, 0x4E);
    
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(min, max, 0xCC);

    s = _mm256_shuffle_epi32(d01, 0xB1);
    
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(min, max, 0xAA);
}

static inline void bitonic_sort_float_01v_descending(__m256& d01) {
    __m256  min, max, s;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_shuffle_epi32(d01, 0xB1);
    
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(max, min, 0xAA);

    s = _mm256_shuffle_epi32(d01, 0x1B);
    
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(max, min, 0xCC);

    s = _mm256_shuffle_epi32(d01, 0xB1);
    
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(max, min, 0xAA);

    s = _mm256_permute4x64_pd((__m256d) _mm256_shuffle_epi32(d01, 0x1B), 0x4E);
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(max, min, 0xF0);

    s = _mm256_shuffle_epi32(d01, 0x4E);
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(max, min, 0xCC);

    s = _mm256_shuffle_epi32(d01, 0xB1);
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(max, min, 0xAA);
}

static inline void bitonic_sort_float_01v_merge_descending(__m256& d01) {
    __m256  min, max, s;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    s = _mm256_permute4x64_pd((__m256d) d01, 0x4E);
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(max, min, 0xF0);

    s = _mm256_shuffle_epi32(d01, 0x4E);
    
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(max, min, 0xCC);

    s = _mm256_shuffle_epi32(d01, 0xB1);
    
    min = _mm256_min_ps(s, d01);
    max = _mm256_max_ps(s, d01);
    d01 = _mm256_blend_epi32(max, min, 0xAA);
}

static inline void bitonic_sort_float_02v_ascending(__m256& d01, __m256& d02) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_01v_ascending(d01);
    bitonic_sort_float_01v_descending(d02);

    tmp = d02;
    
    d02 = _mm256_max_ps(d01, d02);
    d01 = _mm256_min_ps(d01, tmp);

    bitonic_sort_float_01v_merge_ascending(d01);
    bitonic_sort_float_01v_merge_ascending(d02);
}

static inline void bitonic_sort_float_02v_descending(__m256& d01, __m256& d02) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_01v_descending(d01);
    bitonic_sort_float_01v_ascending(d02);

    tmp = d02;
    
    d02 = _mm256_max_ps(d01, d02);
    d01 = _mm256_min_ps(d01, tmp);

    bitonic_sort_float_01v_merge_descending(d01);
    bitonic_sort_float_01v_merge_descending(d02);
}

static inline void bitonic_sort_float_02v_merge_ascending(__m256& d01, __m256& d02) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d02, d01);
    
    d02 = _mm256_max_ps(d02, tmp);

    bitonic_sort_float_01v_merge_ascending(d01);
    bitonic_sort_float_01v_merge_ascending(d02);
}

static inline void bitonic_sort_float_02v_merge_descending(__m256& d01, __m256& d02) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d02, d01);
    
    d02 = _mm256_max_ps(d02, tmp);

    bitonic_sort_float_01v_merge_descending(d01);
    bitonic_sort_float_01v_merge_descending(d02);
}

static inline void bitonic_sort_float_03v_ascending(__m256& d01, __m256& d02, __m256& d03) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_02v_ascending(d01, d02);
    bitonic_sort_float_01v_descending(d03);

    tmp = d03;
    
    d03 = _mm256_max_ps(d02, d03);
    d02 = _mm256_min_ps(d02, tmp);

    bitonic_sort_float_02v_merge_ascending(d01, d02);
    bitonic_sort_float_01v_merge_ascending(d03);
}

static inline void bitonic_sort_float_03v_descending(__m256& d01, __m256& d02, __m256& d03) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_02v_descending(d01, d02);
    bitonic_sort_float_01v_ascending(d03);

    tmp = d03;
    
    d03 = _mm256_max_ps(d02, d03);
    d02 = _mm256_min_ps(d02, tmp);

    bitonic_sort_float_02v_merge_descending(d01, d02);
    bitonic_sort_float_01v_merge_descending(d03);
}

static inline void bitonic_sort_float_03v_merge_ascending(__m256& d01, __m256& d02, __m256& d03) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d03, d01);
    
    d03 = _mm256_max_ps(d03, tmp);

    bitonic_sort_float_02v_merge_ascending(d01, d02);
    bitonic_sort_float_01v_merge_ascending(d03);
}

static inline void bitonic_sort_float_03v_merge_descending(__m256& d01, __m256& d02, __m256& d03) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d03, d01);
    
    d03 = _mm256_max_ps(d03, tmp);

    bitonic_sort_float_02v_merge_descending(d01, d02);
    bitonic_sort_float_01v_merge_descending(d03);
}

static inline void bitonic_sort_float_04v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_02v_ascending(d01, d02);
    bitonic_sort_float_02v_descending(d03, d04);

    tmp = d03;
    
    d03 = _mm256_max_ps(d02, d03);
    d02 = _mm256_min_ps(d02, tmp);

    tmp = d04;
    
    d04 = _mm256_max_ps(d01, d04);
    d01 = _mm256_min_ps(d01, tmp);

    bitonic_sort_float_02v_merge_ascending(d01, d02);
    bitonic_sort_float_02v_merge_ascending(d03, d04);
}

static inline void bitonic_sort_float_04v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_02v_descending(d01, d02);
    bitonic_sort_float_02v_ascending(d03, d04);

    tmp = d03;
    
    d03 = _mm256_max_ps(d02, d03);
    d02 = _mm256_min_ps(d02, tmp);

    tmp = d04;
    
    d04 = _mm256_max_ps(d01, d04);
    d01 = _mm256_min_ps(d01, tmp);

    bitonic_sort_float_02v_merge_descending(d01, d02);
    bitonic_sort_float_02v_merge_descending(d03, d04);
}

static inline void bitonic_sort_float_04v_merge_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d03, d01);
    
    d03 = _mm256_max_ps(d03, tmp);

    tmp = d02;
    
    d02 = _mm256_min_ps(d04, d02);
    
    d04 = _mm256_max_ps(d04, tmp);

    bitonic_sort_float_02v_merge_ascending(d01, d02);
    bitonic_sort_float_02v_merge_ascending(d03, d04);
}

static inline void bitonic_sort_float_04v_merge_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d03, d01);
    
    d03 = _mm256_max_ps(d03, tmp);

    tmp = d02;
    
    d02 = _mm256_min_ps(d04, d02);
    
    d04 = _mm256_max_ps(d04, tmp);

    bitonic_sort_float_02v_merge_descending(d01, d02);
    bitonic_sort_float_02v_merge_descending(d03, d04);
}

static inline void bitonic_sort_float_05v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_float_01v_descending(d05);

    tmp = d05;
    
    d05 = _mm256_max_ps(d04, d05);
    d04 = _mm256_min_ps(d04, tmp);

    bitonic_sort_float_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_float_01v_merge_ascending(d05);
}

static inline void bitonic_sort_float_05v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_04v_descending(d01, d02, d03, d04);
    bitonic_sort_float_01v_ascending(d05);

    tmp = d05;
    
    d05 = _mm256_max_ps(d04, d05);
    d04 = _mm256_min_ps(d04, tmp);

    bitonic_sort_float_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_float_01v_merge_descending(d05);
}

static inline void bitonic_sort_float_05v_merge_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d05, d01);
    
    d05 = _mm256_max_ps(d05, tmp);

    bitonic_sort_float_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_float_01v_merge_ascending(d05);
}

static inline void bitonic_sort_float_05v_merge_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d05, d01);
    
    d05 = _mm256_max_ps(d05, tmp);

    bitonic_sort_float_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_float_01v_merge_descending(d05);
}

static inline void bitonic_sort_float_06v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_float_02v_descending(d05, d06);

    tmp = d05;
    
    d05 = _mm256_max_ps(d04, d05);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_ps(d03, d06);
    d03 = _mm256_min_ps(d03, tmp);

    bitonic_sort_float_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_float_02v_merge_ascending(d05, d06);
}

static inline void bitonic_sort_float_06v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_04v_descending(d01, d02, d03, d04);
    bitonic_sort_float_02v_ascending(d05, d06);

    tmp = d05;
    
    d05 = _mm256_max_ps(d04, d05);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_ps(d03, d06);
    d03 = _mm256_min_ps(d03, tmp);

    bitonic_sort_float_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_float_02v_merge_descending(d05, d06);
}

static inline void bitonic_sort_float_06v_merge_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d05, d01);
    
    d05 = _mm256_max_ps(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_ps(d06, d02);
    
    d06 = _mm256_max_ps(d06, tmp);

    bitonic_sort_float_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_float_02v_merge_ascending(d05, d06);
}

static inline void bitonic_sort_float_06v_merge_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d05, d01);
    
    d05 = _mm256_max_ps(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_ps(d06, d02);
    
    d06 = _mm256_max_ps(d06, tmp);

    bitonic_sort_float_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_float_02v_merge_descending(d05, d06);
}

static inline void bitonic_sort_float_07v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_float_03v_descending(d05, d06, d07);

    tmp = d05;
    
    d05 = _mm256_max_ps(d04, d05);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_ps(d03, d06);
    d03 = _mm256_min_ps(d03, tmp);

    tmp = d07;
    
    d07 = _mm256_max_ps(d02, d07);
    d02 = _mm256_min_ps(d02, tmp);

    bitonic_sort_float_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_float_03v_merge_ascending(d05, d06, d07);
}

static inline void bitonic_sort_float_07v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_04v_descending(d01, d02, d03, d04);
    bitonic_sort_float_03v_ascending(d05, d06, d07);

    tmp = d05;
    
    d05 = _mm256_max_ps(d04, d05);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_ps(d03, d06);
    d03 = _mm256_min_ps(d03, tmp);

    tmp = d07;
    
    d07 = _mm256_max_ps(d02, d07);
    d02 = _mm256_min_ps(d02, tmp);

    bitonic_sort_float_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_float_03v_merge_descending(d05, d06, d07);
}

static inline void bitonic_sort_float_07v_merge_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d05, d01);
    
    d05 = _mm256_max_ps(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_ps(d06, d02);
    
    d06 = _mm256_max_ps(d06, tmp);

    tmp = d03;
    
    d03 = _mm256_min_ps(d07, d03);
    
    d07 = _mm256_max_ps(d07, tmp);

    bitonic_sort_float_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_float_03v_merge_ascending(d05, d06, d07);
}

static inline void bitonic_sort_float_07v_merge_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d05, d01);
    
    d05 = _mm256_max_ps(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_ps(d06, d02);
    
    d06 = _mm256_max_ps(d06, tmp);

    tmp = d03;
    
    d03 = _mm256_min_ps(d07, d03);
    
    d07 = _mm256_max_ps(d07, tmp);

    bitonic_sort_float_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_float_03v_merge_descending(d05, d06, d07);
}

static inline void bitonic_sort_float_08v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_04v_ascending(d01, d02, d03, d04);
    bitonic_sort_float_04v_descending(d05, d06, d07, d08);

    tmp = d05;
    
    d05 = _mm256_max_ps(d04, d05);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_ps(d03, d06);
    d03 = _mm256_min_ps(d03, tmp);

    tmp = d07;
    
    d07 = _mm256_max_ps(d02, d07);
    d02 = _mm256_min_ps(d02, tmp);

    tmp = d08;
    
    d08 = _mm256_max_ps(d01, d08);
    d01 = _mm256_min_ps(d01, tmp);

    bitonic_sort_float_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_float_04v_merge_ascending(d05, d06, d07, d08);
}

static inline void bitonic_sort_float_08v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_04v_descending(d01, d02, d03, d04);
    bitonic_sort_float_04v_ascending(d05, d06, d07, d08);

    tmp = d05;
    
    d05 = _mm256_max_ps(d04, d05);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d06;
    
    d06 = _mm256_max_ps(d03, d06);
    d03 = _mm256_min_ps(d03, tmp);

    tmp = d07;
    
    d07 = _mm256_max_ps(d02, d07);
    d02 = _mm256_min_ps(d02, tmp);

    tmp = d08;
    
    d08 = _mm256_max_ps(d01, d08);
    d01 = _mm256_min_ps(d01, tmp);

    bitonic_sort_float_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_float_04v_merge_descending(d05, d06, d07, d08);
}

static inline void bitonic_sort_float_08v_merge_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d05, d01);
    
    d05 = _mm256_max_ps(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_ps(d06, d02);
    
    d06 = _mm256_max_ps(d06, tmp);

    tmp = d03;
    
    d03 = _mm256_min_ps(d07, d03);
    
    d07 = _mm256_max_ps(d07, tmp);

    tmp = d04;
    
    d04 = _mm256_min_ps(d08, d04);
    
    d08 = _mm256_max_ps(d08, tmp);

    bitonic_sort_float_04v_merge_ascending(d01, d02, d03, d04);
    bitonic_sort_float_04v_merge_ascending(d05, d06, d07, d08);
}

static inline void bitonic_sort_float_08v_merge_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    tmp = d01;
    
    d01 = _mm256_min_ps(d05, d01);
    
    d05 = _mm256_max_ps(d05, tmp);

    tmp = d02;
    
    d02 = _mm256_min_ps(d06, d02);
    
    d06 = _mm256_max_ps(d06, tmp);

    tmp = d03;
    
    d03 = _mm256_min_ps(d07, d03);
    
    d07 = _mm256_max_ps(d07, tmp);

    tmp = d04;
    
    d04 = _mm256_min_ps(d08, d04);
    
    d08 = _mm256_max_ps(d08, tmp);

    bitonic_sort_float_04v_merge_descending(d01, d02, d03, d04);
    bitonic_sort_float_04v_merge_descending(d05, d06, d07, d08);
}

static inline void bitonic_sort_float_09v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_01v_descending(d09);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    bitonic_sort_float_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_01v_merge_ascending(d09);
}

static inline void bitonic_sort_float_09v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_01v_ascending(d09);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    bitonic_sort_float_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_01v_merge_descending(d09);
}

static inline void bitonic_sort_float_10v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_02v_descending(d09, d10);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    bitonic_sort_float_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_02v_merge_ascending(d09, d10);
}

static inline void bitonic_sort_float_10v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_02v_ascending(d09, d10);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    bitonic_sort_float_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_02v_merge_descending(d09, d10);
}

static inline void bitonic_sort_float_11v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_03v_descending(d09, d10, d11);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    bitonic_sort_float_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_03v_merge_ascending(d09, d10, d11);
}

static inline void bitonic_sort_float_11v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_03v_ascending(d09, d10, d11);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    bitonic_sort_float_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_03v_merge_descending(d09, d10, d11);
}

static inline void bitonic_sort_float_12v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11, __m256& d12) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_04v_descending(d09, d10, d11, d12);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_ps(d05, d12);
    d05 = _mm256_min_ps(d05, tmp);

    bitonic_sort_float_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_04v_merge_ascending(d09, d10, d11, d12);
}

static inline void bitonic_sort_float_12v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11, __m256& d12) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_04v_ascending(d09, d10, d11, d12);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_ps(d05, d12);
    d05 = _mm256_min_ps(d05, tmp);

    bitonic_sort_float_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_04v_merge_descending(d09, d10, d11, d12);
}

static inline void bitonic_sort_float_13v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11, __m256& d12, __m256& d13) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_05v_descending(d09, d10, d11, d12, d13);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_ps(d05, d12);
    d05 = _mm256_min_ps(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_ps(d04, d13);
    d04 = _mm256_min_ps(d04, tmp);

    bitonic_sort_float_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_05v_merge_ascending(d09, d10, d11, d12, d13);
}

static inline void bitonic_sort_float_13v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11, __m256& d12, __m256& d13) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_05v_ascending(d09, d10, d11, d12, d13);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_ps(d05, d12);
    d05 = _mm256_min_ps(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_ps(d04, d13);
    d04 = _mm256_min_ps(d04, tmp);

    bitonic_sort_float_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_05v_merge_descending(d09, d10, d11, d12, d13);
}

static inline void bitonic_sort_float_14v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11, __m256& d12, __m256& d13, __m256& d14) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_06v_descending(d09, d10, d11, d12, d13, d14);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_ps(d05, d12);
    d05 = _mm256_min_ps(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_ps(d04, d13);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_ps(d03, d14);
    d03 = _mm256_min_ps(d03, tmp);

    bitonic_sort_float_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_06v_merge_ascending(d09, d10, d11, d12, d13, d14);
}

static inline void bitonic_sort_float_14v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11, __m256& d12, __m256& d13, __m256& d14) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_06v_ascending(d09, d10, d11, d12, d13, d14);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_ps(d05, d12);
    d05 = _mm256_min_ps(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_ps(d04, d13);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_ps(d03, d14);
    d03 = _mm256_min_ps(d03, tmp);

    bitonic_sort_float_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_06v_merge_descending(d09, d10, d11, d12, d13, d14);
}

static inline void bitonic_sort_float_15v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11, __m256& d12, __m256& d13, __m256& d14, __m256& d15) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_07v_descending(d09, d10, d11, d12, d13, d14, d15);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_ps(d05, d12);
    d05 = _mm256_min_ps(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_ps(d04, d13);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_ps(d03, d14);
    d03 = _mm256_min_ps(d03, tmp);

    tmp = d15;
    
    d15 = _mm256_max_ps(d02, d15);
    d02 = _mm256_min_ps(d02, tmp);

    bitonic_sort_float_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_07v_merge_ascending(d09, d10, d11, d12, d13, d14, d15);
}

static inline void bitonic_sort_float_15v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11, __m256& d12, __m256& d13, __m256& d14, __m256& d15) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_07v_ascending(d09, d10, d11, d12, d13, d14, d15);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_ps(d05, d12);
    d05 = _mm256_min_ps(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_ps(d04, d13);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_ps(d03, d14);
    d03 = _mm256_min_ps(d03, tmp);

    tmp = d15;
    
    d15 = _mm256_max_ps(d02, d15);
    d02 = _mm256_min_ps(d02, tmp);

    bitonic_sort_float_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_07v_merge_descending(d09, d10, d11, d12, d13, d14, d15);
}

static inline void bitonic_sort_float_16v_ascending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11, __m256& d12, __m256& d13, __m256& d14, __m256& d15, __m256& d16) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_08v_descending(d09, d10, d11, d12, d13, d14, d15, d16);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_ps(d05, d12);
    d05 = _mm256_min_ps(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_ps(d04, d13);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_ps(d03, d14);
    d03 = _mm256_min_ps(d03, tmp);

    tmp = d15;
    
    d15 = _mm256_max_ps(d02, d15);
    d02 = _mm256_min_ps(d02, tmp);

    tmp = d16;
    
    d16 = _mm256_max_ps(d01, d16);
    d01 = _mm256_min_ps(d01, tmp);

    bitonic_sort_float_08v_merge_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_08v_merge_ascending(d09, d10, d11, d12, d13, d14, d15, d16);
}

static inline void bitonic_sort_float_16v_descending(__m256& d01, __m256& d02, __m256& d03, __m256& d04, __m256& d05, __m256& d06, __m256& d07, __m256& d08, __m256& d09, __m256& d10, __m256& d11, __m256& d12, __m256& d13, __m256& d14, __m256& d15, __m256& d16) {
    __m256  tmp;
    __m256 topBit = _mm256_set1_epi64x(1LLU << 63);

    bitonic_sort_float_08v_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_08v_ascending(d09, d10, d11, d12, d13, d14, d15, d16);

    tmp = d09;
    
    d09 = _mm256_max_ps(d08, d09);
    d08 = _mm256_min_ps(d08, tmp);

    tmp = d10;
    
    d10 = _mm256_max_ps(d07, d10);
    d07 = _mm256_min_ps(d07, tmp);

    tmp = d11;
    
    d11 = _mm256_max_ps(d06, d11);
    d06 = _mm256_min_ps(d06, tmp);

    tmp = d12;
    
    d12 = _mm256_max_ps(d05, d12);
    d05 = _mm256_min_ps(d05, tmp);

    tmp = d13;
    
    d13 = _mm256_max_ps(d04, d13);
    d04 = _mm256_min_ps(d04, tmp);

    tmp = d14;
    
    d14 = _mm256_max_ps(d03, d14);
    d03 = _mm256_min_ps(d03, tmp);

    tmp = d15;
    
    d15 = _mm256_max_ps(d02, d15);
    d02 = _mm256_min_ps(d02, tmp);

    tmp = d16;
    
    d16 = _mm256_max_ps(d01, d16);
    d01 = _mm256_min_ps(d01, tmp);

    bitonic_sort_float_08v_merge_descending(d01, d02, d03, d04, d05, d06, d07, d08);
    bitonic_sort_float_08v_merge_descending(d09, d10, d11, d12, d13, d14, d15, d16);
}

static __attribute__((noinline)) void bitonic_sort_float_01v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    bitonic_sort_float_01v_ascending(d01);
    _mm256_storeu_ps((float *) ptr + 0, d01);
}

static __attribute__((noinline)) void bitonic_sort_float_02v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    bitonic_sort_float_02v_ascending(d01, d02);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
}

static __attribute__((noinline)) void bitonic_sort_float_03v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    bitonic_sort_float_03v_ascending(d01, d02, d03);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
}

static __attribute__((noinline)) void bitonic_sort_float_04v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    bitonic_sort_float_04v_ascending(d01, d02, d03, d04);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
}

static __attribute__((noinline)) void bitonic_sort_float_05v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    bitonic_sort_float_05v_ascending(d01, d02, d03, d04, d05);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
}

static __attribute__((noinline)) void bitonic_sort_float_06v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    __m256 d06 = _mm256_loadu_ps((float const *) ptr + 5);
    bitonic_sort_float_06v_ascending(d01, d02, d03, d04, d05, d06);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
    _mm256_storeu_ps((float *) ptr + 5, d06);
}

static __attribute__((noinline)) void bitonic_sort_float_07v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    __m256 d06 = _mm256_loadu_ps((float const *) ptr + 5);
    __m256 d07 = _mm256_loadu_ps((float const *) ptr + 6);
    bitonic_sort_float_07v_ascending(d01, d02, d03, d04, d05, d06, d07);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
    _mm256_storeu_ps((float *) ptr + 5, d06);
    _mm256_storeu_ps((float *) ptr + 6, d07);
}

static __attribute__((noinline)) void bitonic_sort_float_08v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    __m256 d06 = _mm256_loadu_ps((float const *) ptr + 5);
    __m256 d07 = _mm256_loadu_ps((float const *) ptr + 6);
    __m256 d08 = _mm256_loadu_ps((float const *) ptr + 7);
    bitonic_sort_float_08v_ascending(d01, d02, d03, d04, d05, d06, d07, d08);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
    _mm256_storeu_ps((float *) ptr + 5, d06);
    _mm256_storeu_ps((float *) ptr + 6, d07);
    _mm256_storeu_ps((float *) ptr + 7, d08);
}

static __attribute__((noinline)) void bitonic_sort_float_09v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    __m256 d06 = _mm256_loadu_ps((float const *) ptr + 5);
    __m256 d07 = _mm256_loadu_ps((float const *) ptr + 6);
    __m256 d08 = _mm256_loadu_ps((float const *) ptr + 7);
    __m256 d09 = _mm256_loadu_ps((float const *) ptr + 8);
    bitonic_sort_float_09v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
    _mm256_storeu_ps((float *) ptr + 5, d06);
    _mm256_storeu_ps((float *) ptr + 6, d07);
    _mm256_storeu_ps((float *) ptr + 7, d08);
    _mm256_storeu_ps((float *) ptr + 8, d09);
}

static __attribute__((noinline)) void bitonic_sort_float_10v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    __m256 d06 = _mm256_loadu_ps((float const *) ptr + 5);
    __m256 d07 = _mm256_loadu_ps((float const *) ptr + 6);
    __m256 d08 = _mm256_loadu_ps((float const *) ptr + 7);
    __m256 d09 = _mm256_loadu_ps((float const *) ptr + 8);
    __m256 d10 = _mm256_loadu_ps((float const *) ptr + 9);
    bitonic_sort_float_10v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
    _mm256_storeu_ps((float *) ptr + 5, d06);
    _mm256_storeu_ps((float *) ptr + 6, d07);
    _mm256_storeu_ps((float *) ptr + 7, d08);
    _mm256_storeu_ps((float *) ptr + 8, d09);
    _mm256_storeu_ps((float *) ptr + 9, d10);
}

static __attribute__((noinline)) void bitonic_sort_float_11v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    __m256 d06 = _mm256_loadu_ps((float const *) ptr + 5);
    __m256 d07 = _mm256_loadu_ps((float const *) ptr + 6);
    __m256 d08 = _mm256_loadu_ps((float const *) ptr + 7);
    __m256 d09 = _mm256_loadu_ps((float const *) ptr + 8);
    __m256 d10 = _mm256_loadu_ps((float const *) ptr + 9);
    __m256 d11 = _mm256_loadu_ps((float const *) ptr + 10);
    bitonic_sort_float_11v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
    _mm256_storeu_ps((float *) ptr + 5, d06);
    _mm256_storeu_ps((float *) ptr + 6, d07);
    _mm256_storeu_ps((float *) ptr + 7, d08);
    _mm256_storeu_ps((float *) ptr + 8, d09);
    _mm256_storeu_ps((float *) ptr + 9, d10);
    _mm256_storeu_ps((float *) ptr + 10, d11);
}

static __attribute__((noinline)) void bitonic_sort_float_12v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    __m256 d06 = _mm256_loadu_ps((float const *) ptr + 5);
    __m256 d07 = _mm256_loadu_ps((float const *) ptr + 6);
    __m256 d08 = _mm256_loadu_ps((float const *) ptr + 7);
    __m256 d09 = _mm256_loadu_ps((float const *) ptr + 8);
    __m256 d10 = _mm256_loadu_ps((float const *) ptr + 9);
    __m256 d11 = _mm256_loadu_ps((float const *) ptr + 10);
    __m256 d12 = _mm256_loadu_ps((float const *) ptr + 11);
    bitonic_sort_float_12v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
    _mm256_storeu_ps((float *) ptr + 5, d06);
    _mm256_storeu_ps((float *) ptr + 6, d07);
    _mm256_storeu_ps((float *) ptr + 7, d08);
    _mm256_storeu_ps((float *) ptr + 8, d09);
    _mm256_storeu_ps((float *) ptr + 9, d10);
    _mm256_storeu_ps((float *) ptr + 10, d11);
    _mm256_storeu_ps((float *) ptr + 11, d12);
}

static __attribute__((noinline)) void bitonic_sort_float_13v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    __m256 d06 = _mm256_loadu_ps((float const *) ptr + 5);
    __m256 d07 = _mm256_loadu_ps((float const *) ptr + 6);
    __m256 d08 = _mm256_loadu_ps((float const *) ptr + 7);
    __m256 d09 = _mm256_loadu_ps((float const *) ptr + 8);
    __m256 d10 = _mm256_loadu_ps((float const *) ptr + 9);
    __m256 d11 = _mm256_loadu_ps((float const *) ptr + 10);
    __m256 d12 = _mm256_loadu_ps((float const *) ptr + 11);
    __m256 d13 = _mm256_loadu_ps((float const *) ptr + 12);
    bitonic_sort_float_13v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
    _mm256_storeu_ps((float *) ptr + 5, d06);
    _mm256_storeu_ps((float *) ptr + 6, d07);
    _mm256_storeu_ps((float *) ptr + 7, d08);
    _mm256_storeu_ps((float *) ptr + 8, d09);
    _mm256_storeu_ps((float *) ptr + 9, d10);
    _mm256_storeu_ps((float *) ptr + 10, d11);
    _mm256_storeu_ps((float *) ptr + 11, d12);
    _mm256_storeu_ps((float *) ptr + 12, d13);
}

static __attribute__((noinline)) void bitonic_sort_float_14v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    __m256 d06 = _mm256_loadu_ps((float const *) ptr + 5);
    __m256 d07 = _mm256_loadu_ps((float const *) ptr + 6);
    __m256 d08 = _mm256_loadu_ps((float const *) ptr + 7);
    __m256 d09 = _mm256_loadu_ps((float const *) ptr + 8);
    __m256 d10 = _mm256_loadu_ps((float const *) ptr + 9);
    __m256 d11 = _mm256_loadu_ps((float const *) ptr + 10);
    __m256 d12 = _mm256_loadu_ps((float const *) ptr + 11);
    __m256 d13 = _mm256_loadu_ps((float const *) ptr + 12);
    __m256 d14 = _mm256_loadu_ps((float const *) ptr + 13);
    bitonic_sort_float_14v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
    _mm256_storeu_ps((float *) ptr + 5, d06);
    _mm256_storeu_ps((float *) ptr + 6, d07);
    _mm256_storeu_ps((float *) ptr + 7, d08);
    _mm256_storeu_ps((float *) ptr + 8, d09);
    _mm256_storeu_ps((float *) ptr + 9, d10);
    _mm256_storeu_ps((float *) ptr + 10, d11);
    _mm256_storeu_ps((float *) ptr + 11, d12);
    _mm256_storeu_ps((float *) ptr + 12, d13);
    _mm256_storeu_ps((float *) ptr + 13, d14);
}

static __attribute__((noinline)) void bitonic_sort_float_15v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    __m256 d06 = _mm256_loadu_ps((float const *) ptr + 5);
    __m256 d07 = _mm256_loadu_ps((float const *) ptr + 6);
    __m256 d08 = _mm256_loadu_ps((float const *) ptr + 7);
    __m256 d09 = _mm256_loadu_ps((float const *) ptr + 8);
    __m256 d10 = _mm256_loadu_ps((float const *) ptr + 9);
    __m256 d11 = _mm256_loadu_ps((float const *) ptr + 10);
    __m256 d12 = _mm256_loadu_ps((float const *) ptr + 11);
    __m256 d13 = _mm256_loadu_ps((float const *) ptr + 12);
    __m256 d14 = _mm256_loadu_ps((float const *) ptr + 13);
    __m256 d15 = _mm256_loadu_ps((float const *) ptr + 14);
    bitonic_sort_float_15v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
    _mm256_storeu_ps((float *) ptr + 5, d06);
    _mm256_storeu_ps((float *) ptr + 6, d07);
    _mm256_storeu_ps((float *) ptr + 7, d08);
    _mm256_storeu_ps((float *) ptr + 8, d09);
    _mm256_storeu_ps((float *) ptr + 9, d10);
    _mm256_storeu_ps((float *) ptr + 10, d11);
    _mm256_storeu_ps((float *) ptr + 11, d12);
    _mm256_storeu_ps((float *) ptr + 12, d13);
    _mm256_storeu_ps((float *) ptr + 13, d14);
    _mm256_storeu_ps((float *) ptr + 14, d15);
}

static __attribute__((noinline)) void bitonic_sort_float_16v(float *ptr) {
    __m256 d01 = _mm256_loadu_ps((float const *) ptr + 0);
    __m256 d02 = _mm256_loadu_ps((float const *) ptr + 1);
    __m256 d03 = _mm256_loadu_ps((float const *) ptr + 2);
    __m256 d04 = _mm256_loadu_ps((float const *) ptr + 3);
    __m256 d05 = _mm256_loadu_ps((float const *) ptr + 4);
    __m256 d06 = _mm256_loadu_ps((float const *) ptr + 5);
    __m256 d07 = _mm256_loadu_ps((float const *) ptr + 6);
    __m256 d08 = _mm256_loadu_ps((float const *) ptr + 7);
    __m256 d09 = _mm256_loadu_ps((float const *) ptr + 8);
    __m256 d10 = _mm256_loadu_ps((float const *) ptr + 9);
    __m256 d11 = _mm256_loadu_ps((float const *) ptr + 10);
    __m256 d12 = _mm256_loadu_ps((float const *) ptr + 11);
    __m256 d13 = _mm256_loadu_ps((float const *) ptr + 12);
    __m256 d14 = _mm256_loadu_ps((float const *) ptr + 13);
    __m256 d15 = _mm256_loadu_ps((float const *) ptr + 14);
    __m256 d16 = _mm256_loadu_ps((float const *) ptr + 15);
    bitonic_sort_float_16v_ascending(d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16);
    _mm256_storeu_ps((float *) ptr + 0, d01);
    _mm256_storeu_ps((float *) ptr + 1, d02);
    _mm256_storeu_ps((float *) ptr + 2, d03);
    _mm256_storeu_ps((float *) ptr + 3, d04);
    _mm256_storeu_ps((float *) ptr + 4, d05);
    _mm256_storeu_ps((float *) ptr + 5, d06);
    _mm256_storeu_ps((float *) ptr + 6, d07);
    _mm256_storeu_ps((float *) ptr + 7, d08);
    _mm256_storeu_ps((float *) ptr + 8, d09);
    _mm256_storeu_ps((float *) ptr + 9, d10);
    _mm256_storeu_ps((float *) ptr + 10, d11);
    _mm256_storeu_ps((float *) ptr + 11, d12);
    _mm256_storeu_ps((float *) ptr + 12, d13);
    _mm256_storeu_ps((float *) ptr + 13, d14);
    _mm256_storeu_ps((float *) ptr + 14, d15);
    _mm256_storeu_ps((float *) ptr + 15, d16);
}

static void bitonic_sort_float(float *ptr, int length) {
    const int N = 8;

    switch(length / N) {
        case 1: bitonic_sort_float_01v(ptr); break;
        case 2: bitonic_sort_float_02v(ptr); break;
        case 3: bitonic_sort_float_03v(ptr); break;
        case 4: bitonic_sort_float_04v(ptr); break;
        case 5: bitonic_sort_float_05v(ptr); break;
        case 6: bitonic_sort_float_06v(ptr); break;
        case 7: bitonic_sort_float_07v(ptr); break;
        case 8: bitonic_sort_float_08v(ptr); break;
        case 9: bitonic_sort_float_09v(ptr); break;
        case 10: bitonic_sort_float_10v(ptr); break;
        case 11: bitonic_sort_float_11v(ptr); break;
        case 12: bitonic_sort_float_12v(ptr); break;
        case 13: bitonic_sort_float_13v(ptr); break;
        case 14: bitonic_sort_float_14v(ptr); break;
        case 15: bitonic_sort_float_15v(ptr); break;
        case 16: bitonic_sort_float_16v(ptr); break;
    }
}
}
}
