#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../fixtures.h"

#include <smallsort/bitonic_sort.AVX2.int32_t.generated.h>
#include <smallsort/bitonic_sort.AVX512.int32_t.generated.h>

#include <algorithm>
#include <iterator>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vector_machine;

struct BitonicAVX2_i32 : public SortTest<int32_t> {};

struct BitonicAVX512_i32 : public SortTest<int32_t> {};


INSTANTIATE_TEST_SUITE_P(Bitonic,
                         BitonicAVX2_i32,
                         ValuesIn(range(8, 128, 8)),
                         PrintValue());

TEST_P(BitonicAVX2_i32, BitonicSortAVX2) {
    auto begin = V.data();
    gcsort::smallsort::bitonic<int32_t, gcsort::AVX2>::sort(begin, GetParam());
    EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

INSTANTIATE_TEST_SUITE_P(Bitonic,
                         BitonicAVX512_i32,
                         ValuesIn(range(16, 256, 16)),
                         PrintValue());

TEST_P(BitonicAVX512_i32, BitonicSortAVX512) {
  auto begin = V.data();
  gcsort::smallsort::bitonic<int32_t, gcsort::AVX512>::sort(begin, GetParam());
  EXPECT_THAT(V, WhenSorted(ElementsAreArray(V)));
}

}