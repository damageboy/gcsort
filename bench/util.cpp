#include "util.h"

#include <benchmark/benchmark.h>

namespace vxsort_bench {
Counter make_time_per_n_counter(int64_t n) {
  return { (double)n,
           Counter::kAvgThreadsRate |
               Counter::kIsIterationInvariantRate |
               Counter::kInvert,
        Counter::kIs1000 };
}

Counter make_cycle_per_n_counter(double n) {
  return { n,
           Counter::kDefaults,
           Counter::kIs1000};
}

}
