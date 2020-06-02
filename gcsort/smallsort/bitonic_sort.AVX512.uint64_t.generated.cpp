#include "bitonic_sort.AVX512.uint64_t.generated.h"

using namespace gcsort;

void gcsort::smallsort::bitonic<uint64_t, vector_machine::AVX512 >::sort(uint64_t *ptr, size_t length) {
    const int N = 8;

    switch(length / N) {
        case 1: sort_01v(ptr); break;
        case 2: sort_02v(ptr); break;
        case 3: sort_03v(ptr); break;
        case 4: sort_04v(ptr); break;
        case 5: sort_05v(ptr); break;
        case 6: sort_06v(ptr); break;
        case 7: sort_07v(ptr); break;
        case 8: sort_08v(ptr); break;
        case 9: sort_09v(ptr); break;
        case 10: sort_10v(ptr); break;
        case 11: sort_11v(ptr); break;
        case 12: sort_12v(ptr); break;
        case 13: sort_13v(ptr); break;
        case 14: sort_14v(ptr); break;
        case 15: sort_15v(ptr); break;
        case 16: sort_16v(ptr); break;
    }
}
