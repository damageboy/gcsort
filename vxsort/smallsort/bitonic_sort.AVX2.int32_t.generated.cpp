#include "vxsort_targets_enable_avx2.h"
#include "bitonic_sort.AVX2.int32_t.generated.h"

#include <cassert>
#include <cstdio>
#include <algorithm>


using namespace vxsort;

uint64_t closest_pow2(size_t x) {
    return x == 1 ? 1 : 1<<(64-__builtin_clzl(x-1));
}

static INLINE void cross_min_max(__m256i* __restrict const p1, __m256i* __restrict  const p2) {
    __m256i tmp, dl, dr;
    dl = _mm256_lddqu_si256(p1 - 0);
    dr = _mm256_lddqu_si256(p2 + 0);
    tmp = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(dr, 0x1B), 0x4E);
    dr = _mm256_max_epi32(dl, tmp);
    dl = _mm256_min_epi32(dl, tmp);
    _mm256_storeu_si256(p1 + 0, dl);
    _mm256_storeu_si256(p2 + 0, dr);
}

static INLINE void strided_min_max(__m256i* __restrict const p1, __m256i* __restrict  const p2) {
    __m256i tmp, dl, dr;
    dl = _mm256_lddqu_si256(p1 + 0);
    dr = _mm256_lddqu_si256(p2 + 0);
    tmp = dl;
    dl = _mm256_min_epi32(dr, dl);
    dr = _mm256_max_epi32(dr, tmp);
    _mm256_storeu_si256(p1 + 0, dl);
    _mm256_storeu_si256(p2 + 0, dr);
}

void vxsort::smallsort::bitonic<int32_t, vector_machine::AVX2 >::sort(int32_t *ptr, size_t length) {
    assert((length % 4) == 0);

    auto v = length / N;

    auto p_start = (__m256i*)ptr;
    const auto p_end = p_start + v;

    for (auto p = p_start; p < p_end; p += 4) {
        auto d01 = _mm256_lddqu_si256(p + 0);
        auto d02 = _mm256_lddqu_si256(p + 1);
        auto d03 = _mm256_lddqu_si256(p + 2);
        auto d04 = _mm256_lddqu_si256(p + 3);
        sort_04v_ascending(d01, d02, d03, d04);
        _mm256_storeu_si256(p + 0, d01);
        _mm256_storeu_si256(p + 1, d02);
        _mm256_storeu_si256(p + 2, d03);
        _mm256_storeu_si256(p + 3, d04);
    }

    uint64_t max_v;

    max_v = closest_pow2(v);
    for (int i = 8; i <= max_v; i *= 2) {
        for (auto p = p_start; p < p_end; p += i) {
            auto half_stride = i / 2;
            __m256i*  __restrict p1 = p + half_stride - 1;
            __m256i*  __restrict p2 = p + half_stride;
            auto p2_end = std::min(p + i, p_end);
            for (; p2 < p2_end; p1 -= 4, p2 += 4) {
                cross_min_max(p1 - 0, p2 + 0);
                cross_min_max(p1 - 1, p2 + 1);
                cross_min_max(p1 - 2, p2 + 2);
                cross_min_max(p1 - 3, p2 + 3);
            }
        }

//        for (int j = i/2; j >= 8; j /= 2) {
//            //auto depth = _tzcnt_u64(j);
//            for (auto p = p_start; p < p_end; p += j) {
//                auto half_stride = j/2;
//                __m256i*  __restrict p1 = p;
//                __m256i*  __restrict p2 = p + half_stride;
//                auto p2_end = std::min(p + j, p_end);
//                for (; p2 < p2_end; p1 += 4, p2 += 4) {
//                    strided_min_max(p1 + 0, p2 + 0);
//                    strided_min_max(p1 + 1, p2 + 1);
//                    strided_min_max(p1 + 2, p2 + 2);
//                    strided_min_max(p1 + 3, p2 + 3);
//                }
//            }
//        }

        const auto half_i = i /2;
        for (auto p = p_start; p < p_end; p += half_i) {
            auto p2_max = p + half_i;
            auto p2_end = std::min(p2_max, p_end);
            for (int j = half_i; j >= 8; j /= 2) {
                //auto depth = _tzcnt_u64(j);
                auto half_stride = j/2;
                __m256i*  __restrict p1 = p;
                __m256i*  __restrict p2 = p + half_stride;
                for (; p2 < p2_end; p1 += 4, p2 += 4) {
                    strided_min_max(p1 + 0, p2 + 0);
                    strided_min_max(p1 + 1, p2 + 1);
                    strided_min_max(p1 + 2, p2 + 2);
                    strided_min_max(p1 + 3, p2 + 3);
                }
            }
        }


        for (auto p = p_start; p < p_end; p += 4) {
            auto d01 = _mm256_lddqu_si256(p + 0);
            auto d02 = _mm256_lddqu_si256(p + 1);
            auto d03 = _mm256_lddqu_si256(p + 2);
            auto d04 = _mm256_lddqu_si256(p + 3);
            sort_04v_merge_ascending(d01, d02, d03, d04);
            _mm256_storeu_si256(p + 0, d01);
            _mm256_storeu_si256(p + 1, d02);
            _mm256_storeu_si256(p + 2, d03);
            _mm256_storeu_si256(p + 3, d04);
        }
    }
}

#include "vxsort_targets_disable.h"