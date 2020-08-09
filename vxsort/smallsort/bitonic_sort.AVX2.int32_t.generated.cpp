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
    // We keep up to the last 4 vectors
    // in this temp space, because we need to deal with inputs < 4 vectors
    // and also deal with the tail of the array when it isn't exactly divisible by 4
    // vectors
    __m256i slack[4];

    // Full vectors
    auto v = length / N;
    // # elements in the last vector
    const int remainder = (int) (length - v * N);
    const auto slack_v = (v % 4) + ((remainder > 0) ? 1 : 0);

    assert(remainder * N < sizeof(mask_table_8));
    // Load/Store mask for the last vector
    const auto mask = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(mask_table_8 + remainder * N)));
    // How many vectors in the last group of up to 4 vectors

    auto last_chunk_v = slack_v;

    const auto p_start = (__m256i*)ptr;
    const auto p_end_inplace = p_start + ((v/4) * 4);
    const auto p_virtual_end = p_end_inplace + ((slack_v > 0) ? 4 : 0);
    v += (remainder > 0) ? 1 : 0;

    auto p_exit_loop = p_end_inplace;

    __m256i *p = p_start;
    __m256i d01, d02, d03, d04;

    for (; p < p_exit_loop; p += 4) {
        d01 = _mm256_lddqu_si256(p + 0);
        d02 = _mm256_lddqu_si256(p + 1);
        d03 = _mm256_lddqu_si256(p + 2);
        d04 = _mm256_lddqu_si256(p + 3);
ugly_hack_1:
        sort_04v_ascending(d01, d02, d03, d04);
        _mm256_storeu_si256(p + 0, d01);
        _mm256_storeu_si256(p + 1, d02);
        _mm256_storeu_si256(p + 2, d03);
        _mm256_storeu_si256(p + 3, d04);
    }

    // We jump into the middle of the loop just for one last-time
    // to handle the schwanz
    p_exit_loop = nullptr;
    switch (last_chunk_v) {
        case 1:
            d04 = d03 = d02 = _mm256_set1_epi32(MAX);
            d01 = _mm256_or_si256(_mm256_maskload_epi32((int32_t *) (p + 0), mask),
                                  _mm256_andnot_si256(mask, d02));
            last_chunk_v = 0; p = slack; goto ugly_hack_1;
        case 2:
            d04 = d03 = _mm256_set1_epi32(MAX);
            d01 = _mm256_lddqu_si256(p + 0);
            d02 = _mm256_or_si256(_mm256_maskload_epi32((int32_t *) (p + 1), mask),
                                  _mm256_andnot_si256(mask, d03));
            last_chunk_v = 0; p = slack; goto ugly_hack_1;
        case 3:
            d04 = _mm256_set1_epi32(MAX);
            d01 = _mm256_lddqu_si256(p + 0);
            d02 = _mm256_lddqu_si256(p + 1);
            d03 = _mm256_or_si256(_mm256_maskload_epi32((int32_t *) (p + 2), mask),
                                  _mm256_andnot_si256(mask, d04));

            last_chunk_v = 0; p = slack; goto ugly_hack_1;
        case 4:
            d01 = _mm256_lddqu_si256(p + 0);
            d02 = _mm256_lddqu_si256(p + 1);
            d03 = _mm256_lddqu_si256(p + 2);
            d04 = _mm256_or_si256(_mm256_maskload_epi32((int32_t *) (p + 3), mask),
                                  _mm256_andnot_si256(mask, _mm256_set1_epi32(MAX)));
            last_chunk_v = 0; p = slack; goto ugly_hack_1;
    }
    last_chunk_v = slack_v;
    p_exit_loop = p_end_inplace;

    __m256i*  __restrict p1;
    __m256i*  __restrict p2;
    __m256i *p2_end;
    int half_stride;

    const auto max_v = closest_pow2(v);
    for (int i = 8; i <= max_v; i *= 2) {
        for (p = p_start; p < p_virtual_end; p += i) {
            half_stride = i / 2;
            p1 = p + half_stride - 1;
            p2 = p + half_stride;
            p2_end = std::min(p + i, p_virtual_end);
            for (; p2 < p2_end; p1 -= 4, p2 += 4) {
                if (p2 >= p_end_inplace) {
                    p2 = slack;
                    p2_end = slack + 4;
                }
                cross_min_max(p1 - 0, p2 + 0);
                cross_min_max(p1 - 1, p2 + 1);
                cross_min_max(p1 - 2, p2 + 2);
                cross_min_max(p1 - 3, p2 + 3);
            }
        }

        const auto half_i = i /2;

        for (int j = half_i; j >= 8; j /= 2) {
            const auto half_stride = j/2;
            for (auto p = p_start; p < p_virtual_end; p += j) {
                p1 = p;
                p2 = p + half_stride;
                p2_end = std::min(p + j, p_virtual_end);
                for (; p2 < p2_end; p1 += 4, p2 += 4) {
                    if (p2 >= p_end_inplace) {
                        p2 = slack;
                        p2_end = slack + 4;
                    }
                    strided_min_max(p1 + 0, p2 + 0);
                    strided_min_max(p1 + 1, p2 + 1);
                    strided_min_max(p1 + 2, p2 + 2);
                    strided_min_max(p1 + 3, p2 + 3);
                }
            }
        }
        p = p_start;
        for (; p < p_exit_loop; p += 4) {
ugly_hack_4:
            d01 = _mm256_lddqu_si256(p + 0);
            d02 = _mm256_lddqu_si256(p + 1);
            d03 = _mm256_lddqu_si256(p + 2);
            d04 = _mm256_lddqu_si256(p + 3);
            sort_04v_merge_ascending(d01, d02, d03, d04);
            _mm256_storeu_si256(p + 0, d01);
            _mm256_storeu_si256(p + 1, d02);
            _mm256_storeu_si256(p + 2, d03);
            _mm256_storeu_si256(p + 3, d04);
        }
        if (last_chunk_v > 0) {
            p = slack;
            p_exit_loop = nullptr;
            last_chunk_v = 0;
            goto ugly_hack_4;
        }
        last_chunk_v = slack_v;
        p_exit_loop = p_end_inplace;

    }

    switch (last_chunk_v) {
        case 0:
            break;
        case 1:
            d01 = _mm256_lddqu_si256(slack + 0);
            _mm256_maskstore_epi32((int32_t *) (p_end_inplace + 0), mask, d01);
            break;
        case 2:
            d01 = _mm256_lddqu_si256(slack + 0);
            d02 = _mm256_lddqu_si256(slack + 1);
            _mm256_storeu_si256(p_end_inplace + 0, d01);
            _mm256_maskstore_epi32((int32_t *) (p_end_inplace + 1), mask, d02);
            break;
        case 3:
            d01 = _mm256_lddqu_si256(slack + 0);
            d02 = _mm256_lddqu_si256(slack + 1);
            d03 = _mm256_lddqu_si256(slack + 2);
            _mm256_storeu_si256(p_end_inplace + 0, d01);
            _mm256_storeu_si256(p_end_inplace + 1, d02);
            _mm256_maskstore_epi32((int32_t *) (p_end_inplace + 2), mask, d03);
            break;
        case 4:
            d01 = _mm256_lddqu_si256(slack + 0);
            d02 = _mm256_lddqu_si256(slack + 1);
            d03 = _mm256_lddqu_si256(slack + 2);
            d04 = _mm256_lddqu_si256(slack + 3);
            _mm256_storeu_si256(p_end_inplace + 0, d01);
            _mm256_storeu_si256(p_end_inplace + 1, d02);
            _mm256_storeu_si256(p_end_inplace + 2, d03);
            _mm256_maskstore_epi32((int32_t *) (p_end_inplace + 3), mask, d04);
            break;
    }


}

#include "vxsort_targets_disable.h"
#pragma clang diagnostic pop
