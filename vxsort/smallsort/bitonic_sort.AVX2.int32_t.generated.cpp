#pragma clang diagnostic push
#include "vxsort_targets_enable_avx2.h"
#include "bitonic_sort.AVX2.int32_t.generated.h"

#include <cassert>
#include <cstdio>
#include <algorithm>


using namespace vxsort;

uint64_t closest_pow2(size_t x) {
    return x == 1U ? 1U : 1U<<(64-__builtin_clzl(x-1));
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
    //const int remainder = (int) (length - v * N);
    //assert(remainder * N < sizeof(mask_table_8));
    //const auto mask = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(mask_table_8 + remainder * N)));
    //const auto last_chunk_v = (v % 4) + ((remainder > 0) ? 1 : 0);

    auto p_start = (__m256i*)ptr;
    const auto p_end = p_start + v;
    //const auto p_end_quads = p_end - last_chunk_v;

    //__m256i d01 = _mm256_lddqu_si256((__m256i const *) ptr + 0);
    //__m256i d02 = _mm256_lddqu_si256((__m256i const *) ptr + 1);
    //__m256i d03 = _mm256_lddqu_si256((__m256i const *) ptr + 2);
    //__m256i d04 = _mm256_or_si256(_mm256_maskload_epi32((int32_t const *) ((__m256i const *) ptr + 3), mask), _mm256_andnot_si256(mask, _mm256_set1_epi32(MAX)));


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

        const auto half_i = i /2;
//#define OLD_WAY
#ifdef OLD_WAY
        for (int j = half_i; j >= 8; j /= 2) {
            const auto half_stride = j/2;
            for (auto p = p_start; p < p_end; p += j) {
                __m256i*  __restrict p1 = p;
                __m256i*  __restrict p2 = p + half_stride;
                auto p2_end = std::min(p + j, p_end);
                for (; p2 < p2_end; p1 += 4, p2 += 4) {
                    strided_min_max(p1 + 0, p2 + 0);
                    strided_min_max(p1 + 1, p2 + 1);
                    strided_min_max(p1 + 2, p2 + 2);
                    strided_min_max(p1 + 3, p2 + 3);
                }
            }
        }
#else
        for (auto p = p_start; p < p_end; p += half_i) {
            auto p2_max = p + half_i;
            auto p2_end = std::min(p2_max, p_end);
            for (int j = half_i; j >= 8; j /= 2) {

                auto half_stride = j/2;
//                auto depth = _tzcnt_u64(j);
//                const auto length = p2_end - p;
//                if (depth < 5 || length != 32) {
                    __m256i* __restrict p1 = p;
                    __m256i* __restrict p2 = p + half_stride;
                    // printf("min-max: %llu, <->  %ld\n", depth, p2_end - p1);
                    for (; p2 < p2_end; p1 += 4, p2 += 4) {
                        strided_min_max(p1 + 0, p2 + 0);
                        strided_min_max(p1 + 1, p2 + 1);
                        strided_min_max(p1 + 2, p2 + 2);
                        strided_min_max(p1 + 3, p2 + 3);
                    }
                    //j /= 2;
#ifdef XXX
                } else {
                    __m256i* __restrict p1 = p + 0;
                    __m256i* __restrict p2 = p + 16;
                    // printf("3x min-max: %llu, <->  %ld\n", depth, p2_end - p1);

                    for (int x = 0; x < 4; p1++, p2++, x++) {
                        __m256i tmp;
                        auto dl01 = _mm256_lddqu_si256(p1 + 0);
                        auto dl02 = _mm256_lddqu_si256(p1 + 4);
                        auto dl03 = _mm256_lddqu_si256(p1 + 8);
                        auto dl04 = _mm256_lddqu_si256(p1 + 12);

                        auto dr01 = _mm256_lddqu_si256(p2 + 0);
                        auto dr02 = _mm256_lddqu_si256(p2 + 4);
                        auto dr03 = _mm256_lddqu_si256(p2 + 8);
                        auto dr04 = _mm256_lddqu_si256(p2 + 12);
                        // 1
                        tmp = dl01;
                        dl01 = _mm256_min_epi32(dr01, dl01);
                        dr01 = _mm256_max_epi32(dr01, tmp);
                        tmp = dl02;
                        dl02 = _mm256_min_epi32(dr02, dl02);
                        dr02 = _mm256_max_epi32(dr02, tmp);
                        tmp = dl03;
                        dl03 = _mm256_min_epi32(dr03, dl03);
                        dr03 = _mm256_max_epi32(dr03, tmp);
                        tmp = dl04;
                        dl04 = _mm256_min_epi32(dr04, dl04);
                        dr04 = _mm256_max_epi32(dr04, tmp);
                        // 2
                        tmp = dl01;
                        dl01 = _mm256_min_epi32(dl03, dl01);
                        dl03 = _mm256_max_epi32(dl03, tmp);
                        tmp = dl02;
                        dl02 = _mm256_min_epi32(dl04, dl02);
                        dl04 = _mm256_max_epi32(dl04, tmp);
                        tmp = dr01;
                        dr01 = _mm256_min_epi32(dr03, dr01);
                        dr03 = _mm256_max_epi32(dr03, tmp);
                        tmp = dr02;
                        dr02 = _mm256_min_epi32(dr04, dr02);
                        dr04 = _mm256_max_epi32(dr04, tmp);
                        // 3
                        tmp = dl01;
                        dl01 = _mm256_min_epi32(dl02, dl01);
                        dl02 = _mm256_max_epi32(dl02, tmp);
                        tmp = dl03;
                        dl03 = _mm256_min_epi32(dl04, dl03);
                        dl04 = _mm256_max_epi32(dl04, tmp);
                        tmp = dr01;
                        dr01 = _mm256_min_epi32(dr02, dr01);
                        dr02 = _mm256_max_epi32(dr02, tmp);
                        tmp = dr03;
                        dr03 = _mm256_min_epi32(dr04, dr03);
                        dr04 = _mm256_max_epi32(dr04, tmp);

                        _mm256_storeu_si256(p1 + 0, dl01);
                        _mm256_storeu_si256(p1 + 4, dl02);
                        _mm256_storeu_si256(p1 + 8, dl03);
                        _mm256_storeu_si256(p1 + 12, dl04);

                        _mm256_storeu_si256(p2 + 0, dr01);
                        _mm256_storeu_si256(p2 + 4, dr02);
                        _mm256_storeu_si256(p2 + 8, dr03);
                        _mm256_storeu_si256(p2 + 12, dr04);
                    }
                    j /= 8;
                }
#endif
            }
        }
#endif
        //printf("merge!\n");
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
#pragma clang diagnostic pop