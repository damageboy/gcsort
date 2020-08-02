#include "bitonic_sort.AVX2.int32_t.generated.h"

#include <cassert>
#include <cstdio>

using namespace vxsort;

void vxsort::smallsort::bitonic<int32_t, vector_machine::AVX2 >::sort(int32_t *ptr, size_t length) {
    assert((length & (length -1)) == 0);

    const auto fullvlength = length / N;
    const int remainder = (int)(length - fullvlength * N);
    const auto v = fullvlength + ((remainder > 0) ? 1 : 0);
    switch (v) {
        case 1:
            sort_01v_alt(ptr, remainder);
            return;
        case 2:
            sort_02v_alt(ptr, remainder);
            return;
        case 3:
            sort_03v_alt(ptr, remainder);
            return;
        case 4:
            sort_04v_alt(ptr, remainder);
            return;
    }

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

    const auto max_v = 1 << _tzcnt_u64(v);

    for (int i = 8; i <= max_v; i *= 2) {
        for (auto p = p_start; p < p_end; p += i) {
            auto p1 = p;
            auto p2 = p + i - 1;
            for (int j = 0; j < i / 2; j++) {
                auto dl = _mm256_lddqu_si256(p1);
                auto dr = _mm256_lddqu_si256(p2);

                auto tmp = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(dr, 0x1B), 0x4E);
                dr = _mm256_max_epi32(dl, tmp);
                dl = _mm256_min_epi32(dl, tmp);
                _mm256_storeu_si256(p1, dl);
                _mm256_storeu_si256(p2, dr);
                p1++;
                p2--;
            }
        }

        for (int j = i; j >= 16; j /= 2) {
            for (auto p = p_start; p < p_end; p += j/2) {
                auto p1 = p;
                auto p2 = p + j / 4;
                for (auto x = 0; x < j/4; x++) {
                    auto dl = _mm256_lddqu_si256(p1);
                    auto dr = _mm256_lddqu_si256(p2);
                    auto tmp = dl;
                    dl = _mm256_min_epi32(dr, dl);
                    dr = _mm256_max_epi32(dr, tmp);
                    _mm256_storeu_si256(p1, dl);
                    _mm256_storeu_si256(p2, dr);
                    p1++;
                    p2++;
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

