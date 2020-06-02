#include "gtest/gtest.h"
#include "../fixtures.h"
#include "fullsort.h"

#include "vxsort_targets_enable.h"

#include <vxsort.uint64_t.h>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vector_machine;

struct FullSortTest_ui64 : public SortWithSlackTest<uint64_t> {};

INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_ui64,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_ui64::VectorElements*2)),
                         PrintSizeAndSlack());



TEST_P(FullSortTest_ui64, VxSortAVX2_1)  { perform_vxsort_test<uint64_t,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX2_2)  { perform_vxsort_test<uint64_t,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX2_4)  { perform_vxsort_test<uint64_t,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX2_8)  { perform_vxsort_test<uint64_t,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX2_12) { perform_vxsort_test<uint64_t, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTest_ui64, VxSortAVX512_1)  { perform_vxsort_test<uint64_t,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX512_2)  { perform_vxsort_test<uint64_t,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX512_4)  { perform_vxsort_test<uint64_t,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX512_8)  { perform_vxsort_test<uint64_t,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_ui64, VxSortAVX512_12) { perform_vxsort_test<uint64_t, 12, vector_machine::AVX512>(V); }

}

#include "vxsort_targets_disable.h"