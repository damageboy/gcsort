
#include "vxsort_targets_enable_avx2.h"

#include "smallsort/bitonic_machine.AVX2.int32_t.generated.h"
#include "smallsort/bitonic_sort.h"
#include "vector_machine/machine_traits.avx2.h.h"
#include "vxsort.h"

void do_avx2(int *begin, int *end) {
  auto sorter = vxsort::vxsort<int, vxsort::vector_machine::AVX2, 8>();
  sorter.sort(begin, end);
}
#include "vxsort_targets_disable.h"
