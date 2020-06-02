#ifndef GCSORT_VXSORT_H
#define GCSORT_VXSORT_H

#include "vxsort_targets_enable.h"

#include <assert.h>
#include <immintrin.h>


#include "defs.h"
#include "isa_detection.h"
#include "machine_traits.h"
#include "smallsort/bitonic_sort.h"

#include <algorithm>
#include <cstring>
#include <cstdint>

#ifdef ARCH_X64
//#include "machine_traits.avx2.h"
//#include "machine_traits.avx512.h"
//
// #include "smallsort/bitonic_sort.AVX2.int64_t.generated.h"
//#include "smallsort/bitonic_sort.AVX2.uint64_t.generated.h"
//#include "smallsort/bitonic_sort.AVX2.double.generated.h"
//#include "smallsort/bitonic_sort.AVX2.float.generated.h"
//#include "smallsort/bitonic_sort.AVX2.uint32_t.generated.h"
//
//#include "smallsort/bitonic_sort.AVX512.int64_t.generated.h"
//#include "smallsort/bitonic_sort.AVX512.uint64_t.generated.h"
//#include "smallsort/bitonic_sort.AVX512.double.generated.h"
//#include "smallsort/bitonic_sort.AVX512.float.generated.h"
//#include "smallsort/bitonic_sort.AVX512.uint32_t.generated.h"
#endif

#ifdef ARCH_ARM64
#error "─│─│──╫▓▓▓╫──│─────│─│─│──────╫▓▓╫│──│─│"
#error "──│─▓███████▓─╫╫╫╫╫╫╫╫╫╫╫╫╫│▓███████╫──"
#error "───██████████████████████████████████▓─"
#error "│─████████████│─│─│─│─────▓███████████╫"
#error "╫█████────│───│─│───│─────│─│─│───╫████▓"
#error "│████│──│───│───│─│───│───────│─│─│▓███╫"
#error "─▓███│───────│─▓██───│╫██╫─│─│─│───▓███│"
#error "──███─│──────╫████▓───█████────────▓███─"
#error "──╫██──│─│──╫██████│─│██████─│─────▓██─│"
#error "│─│▓█││─│─││███▓▓██─│─██▓▓███─│─│──▓█─│─"
#error "────█│─│───███╫▓▓█▓│──█▓▓▓▓██▓─────▓█───"
#error "│─││█││───▓███╫██▓╫─│─▓▓█▓▓███─────▓█───"
#error "─│─╫█│─│─│████▓╫▓▓─────█▓╫████▓──│─▓█───"
#error "│─││█╫│─││███████─│██╫│▓███████─│─│██─│─"
#error "─│─│█▓╫╫─▓██████╫│─▓█│──▓██████│╫╫│██│─│"
#error "│─│─██│╫│▓█████╫│───▓───│▓█████╫╫╫╫█▓──"
#error "─│─│▓█╫││╫████╫│││╫██▓││││▓████│╫─▓█╫│─│"
#error "│─│─│██│││╫▓▓││╫╫╫╫╫▓╫╫╫╫╫│╫▓▓╫││╫██──│─"
#error "─│───▓██╫─────││││││─││││││────│▓██│────"
#error "│─│─│─▓██▓╫╫╫╫╫╫╫╫▓▓▓▓▓╫╫╫╫╫╫╫▓███│────"
#error "───────╫██████████▓▓▓▓▓██████████│────│"
#error "│─│─│───▓█████████╫─│─▓█████████│─│─│─│"
#error "─────────██████████──│█████████╫─│───││"
#error "│─│─│───│▓█╫███████││▓███████╫█││─│─│─│"
#error "───────│─██─╫██████▓─███████││█╫───│──│"
#error "│───│───│██─││█████▓─█████▓─│╫█╫│──────"
#error "─│─│───│─▓█──│─╫▓██│─▓██▓│─│─▓█│───────"
#error "│───│─│─│─██────│─│───│─────│██───│─│─│"
#error "─│─│───│─│▓██╫─│─│─────│─│─▓██││─│───│─│"
#error "│───────│─│██████████████████▓│─│─│─│─│"
#error "─│───│─│───│███████▓▓████████│─│───│──│"
#error "│─│───│─│─│─│██████╫─▓█████▓────│─│─│──"
#error "─────│─────╫│╫▓████▓─█████▓│╫╫───────│"
#error "│─│───│───╫─╫╫╫╫███╫╫╫██▓╫│╫╫╫│─│─────"
#endif

namespace gcsort {
using gcsort::smallsort::bitonic;

template <int N>
struct alignment_hint {
public:
    static const size_t ALIGN = N;
    static const int8_t REALIGN = 0x66;

    alignment_hint() : left_align(REALIGN), right_align(REALIGN) {}
    alignment_hint realign_left() {
        alignment_hint copy = *this;
        copy.left_align = REALIGN;
        return copy;
    }

    alignment_hint realign_right() {
        alignment_hint copy = *this;
        copy.right_align = REALIGN;
        return copy;
    }

    static bool is_aligned(void *p) {
        return (size_t)p % ALIGN == 0;
    }

    int left_align : 8;
    int right_align : 8;
};

template <typename T, vector_machine M, int Unroll=1>
class vxsort {
    static_assert(Unroll >= 1, "Unroll can be in the range 1..12");
    static_assert(Unroll <= 12, "Unroll can be in the range 1..12");

private:
    //using Tv2 = Tp::TV;
    using Tp = vxsort_machine_traits<T, M>;
    typedef typename Tp::TV TV;
    typedef alignment_hint<sizeof(TV)> AH;

    static const int ELEMENT_ALIGN = sizeof(T) - 1;
    static const int N = sizeof(TV) / sizeof(T);
    static const int32_t MAX_BITONIC_SORT_VECTORS = 16;
    static const int32_t SMALL_SORT_THRESHOLD_ELEMENTS = MAX_BITONIC_SORT_VECTORS * N;
    static const int32_t MaxInnerUnroll = (MAX_BITONIC_SORT_VECTORS - 3) / 2;
    static const int32_t SafeInnerUnroll = MaxInnerUnroll > Unroll ? Unroll : MaxInnerUnroll;
    static const int32_t SLACK_PER_SIDE_IN_VECTORS = Unroll;
    static const size_t ALIGN = AH::ALIGN;
    static const size_t ALIGN_MASK = ALIGN - 1;

    static const int SLACK_PER_SIDE_IN_ELEMENTS = SLACK_PER_SIDE_IN_VECTORS * N;
    // The formula for figuring out how much temporary space we need for partitioning:
    // 2 x the number of slack elements on each side for the purpose of partitioning in unrolled manner +
    // 2 x amount of maximal bytes needed for alignment (32)
    // one more vector's worth of elements since we write with N-way stores from both ends of the temporary area
    // and we must make sure we do not accidentally over-write from left -> right or vice-versa right on that edge...
    // In other words, while we allocated this much temp memory, the actual amount of elements inside said memory
    // is smaller by 8 elements + 1 for each alignment (max alignment is actually N-1, I just round up to N...)
    // This long sense just means that we over-allocate N+2 elements...
    static const int PARTITION_TMP_SIZE_IN_ELEMENTS =
            (2 * SLACK_PER_SIDE_IN_ELEMENTS + N + 4*N);

    static int floor_log2_plus_one(size_t n) {
        auto result = 0;
        while (n >= 1) {
            result++;
            n /= 2;
        }
        return result;
    }
    static void swap(T* left, T* right) {
        auto tmp = *left;
        *left = *right;
        *right = tmp;
    }
    static void swap_if_greater(T* left, T* right) {
        if (*left <= *right)
            return;
        swap(left, right);
    }

    static void insertion_sort(T* lo, T* hi) {
        for (auto i = lo + 1; i <= hi; i++) {
            auto j = i;
            auto t = *i;
            while ((j > lo) && (t < *(j - 1))) {
                *j = *(j - 1);
                j--;
            }
            *j = t;
        }
    }

    static void heap_sort(T* lo, T* hi) {
        size_t n = hi - lo + 1;
        for (size_t i = n / 2; i >= 1; i--) {
            down_heap(i, n, lo);
        }
        for (size_t i = n; i > 1; i--) {
            swap(lo, lo + i - 1);
            down_heap(1, i - 1, lo);
        }
    }
    static void down_heap(size_t i, size_t n, T* lo) {
        auto d = *(lo + i - 1);
        size_t child;
        while (i <= n / 2) {
            child = 2 * i;
            if (child < n && *(lo + child - 1) < (*(lo + child))) {
                child++;
            }
            if (!(d < *(lo + child - 1))) {
                break;
            }
            *(lo + i - 1) = *(lo + child - 1);
            i = child;
        }
        *(lo + i - 1) = d;
    }

    void reset(T* start, T* end) {
        _depth = 0;
        _startPtr = start;
        _endPtr = end;
    }

    T* _startPtr = nullptr;
    T* _endPtr = nullptr;

    T _temp[PARTITION_TMP_SIZE_IN_ELEMENTS];
    int _depth = 0;

    NOINLINE
    T* align_left_scalar_uncommon(T* read_left, T pivot,
                                  T*& tmp_left, T*& tmp_right) {
        if (((size_t)read_left & ALIGN_MASK) == 0)
            return read_left;

        auto next_align = (T*)(((size_t)read_left + ALIGN) & ~ALIGN_MASK);
        while (read_left < next_align) {
            auto v = *(read_left++);
            if (v <= pivot) {
                *(tmp_left++) = v;
            } else {
                *(--tmp_right) = v;
            }
        }

        return read_left;
    }

    NOINLINE
    T* align_right_scalar_uncommon(T* readRight, T pivot,
                                   T*& tmpLeft, T*& tmpRight) {
        if (((size_t) readRight & ALIGN_MASK) == 0)
            return readRight;

        auto nextAlign = (T *) ((size_t) readRight & ~ALIGN_MASK);
        while (readRight > nextAlign) {
            auto v = *(--readRight);
            if (v <= pivot) {
                *(tmpLeft++) = v;
            } else {
                *(--tmpRight) = v;
            }
        }

        return readRight;
    }

    void sort(T* left, T* right, AH realignHint,
              int depthLimit) {
        auto length = (size_t)(right - left + 1);

        T* mid;
        switch (length) {
            case 0:
            case 1:
                return;
            case 2:
                swap_if_greater(left, right);
                return;
            case 3:
                mid = right - 1;
                swap_if_greater(left, mid);
                swap_if_greater(left, right);
                swap_if_greater(mid, right);
                return;
        }

        // Go to insertion sort below this threshold
        if (length <= SMALL_SORT_THRESHOLD_ELEMENTS) {

            auto nextLength = (length & (N-1)) > 0 ? (length + N) & ~(N-1) : length;

            auto extraSpaceNeeded = nextLength - length;
            auto fakeLeft = left - extraSpaceNeeded;
            if (fakeLeft >= _startPtr) {
                bitonic<T, M>::sort(fakeLeft, nextLength);
            } else {
                insertion_sort(left, right);
            }
            return;
        }

        // Detect a whole bunch of bad cases where partitioning
        // will not do well:
        // 1. Reverse sorted array
        // 2. High degree of repeated values (dutch flag problem, one value)
        if (depthLimit == 0) {
            heap_sort(left, right);
            _depth--;
            return;
        }
        depthLimit--;

        // This is going to be a bit weird:
        // Pre/Post alignment calculations happen here: we prepare hints to the
        // partition function of how much to align and in which direction
        // (pre/post). The motivation to do these calculations here and the actual
        // alignment inside the partitioning code is that here, we can cache those
        // calculations. As we recurse to the left we can reuse the left cached
        // calculation, And when we recurse to the right we reuse the right
        // calculation, so we can avoid re-calculating the same aligned addresses
        // throughout the recursion, at the cost of a minor code complexity
        // Since we branch on the magi values REALIGN_LEFT & REALIGN_RIGHT its safe
        // to assume the we are not torturing the branch predictor.'

        // We use a long as a "struct" to pass on alignment hints to the
        // partitioning By packing 2 32 bit elements into it, as the JIT seem to not
        // do this. In reality  we need more like 2x 4bits for each side, but I
        // don't think there is a real difference'

        if (realignHint.left_align == AH::REALIGN) {
            // Alignment flow:
            // * Calculate pre-alignment on the left
            // * See it would cause us an out-of bounds read
            // * Since we'd like to avoid that, we adjust for post-alignment
            // * No branches since we do branch->arithmetic
            auto preAlignedLeft = reinterpret_cast<T*>(reinterpret_cast<size_t>(left) & ~ALIGN_MASK);
            auto cannotPreAlignLeft = (preAlignedLeft - _startPtr) >> 63;
            realignHint.left_align = (preAlignedLeft - left) + (N & cannotPreAlignLeft);
            assert(realignHint.left_align >= -N && realignHint.left_align <= N);
            assert(AH::is_aligned(left + realignHint.left_align));
        }

        if (realignHint.right_align == AH::REALIGN) {
            // Same as above, but in addition:
            // right is pointing just PAST the last element we intend to partition
            // (it's pointing to where we will store the pivot!) So we calculate alignment based on
            // right - 1
            auto preAlignedRight = reinterpret_cast<T*>(((reinterpret_cast<size_t>(right) - 1) & ~ALIGN_MASK) + ALIGN);
            auto cannotPreAlignRight = (_endPtr - preAlignedRight) >> 63;
            realignHint.right_align = (preAlignedRight - right - (N & cannotPreAlignRight));
            assert(realignHint.right_align >= -N && realignHint.right_align <= N);
            assert(AH::is_aligned(right + realignHint.right_align));
        }

        // Compute median-of-three, of:
        // the first, mid and one before last elements
        mid = left + ((right - left) / 2);
        swap_if_greater(left, mid);
        swap_if_greater(left, right - 1);
        swap_if_greater(mid, right - 1);

        // Pivot is mid, place it in the right hand side
        swap(mid, right);

        auto sep = (length < PARTITION_TMP_SIZE_IN_ELEMENTS) ?
                vectorized_partition<SafeInnerUnroll>(left, right, realignHint) :
                vectorized_partition<Unroll>(left, right, realignHint);



        _depth++;
        sort(left, sep - 2, realignHint.realign_right(), depthLimit);
        sort(sep, right, realignHint.realign_left(), depthLimit);
        _depth--;
    }


    static INLINE void partition_block(TV& dataVec,
                                       const TV& P,
                                       T*& left,
                                       T*& right) {
      if (Tp::supports_compress_writes()) {
        partition_block_with_compress(dataVec, P, left, right);
      } else {
        partition_block_without_compress(dataVec, P, left, right);
      }
    }

    static INLINE void partition_block_without_compress(TV& dataVec,
                                                        const TV& P,
                                                        T*& left,
                                                        T*& right) {
        auto mask = Tp::get_cmpgt_mask(dataVec, P);
        dataVec = Tp::partition_vector(dataVec, mask);
        Tp::store_vec(reinterpret_cast<TV*>(left), dataVec);
        Tp::store_vec(reinterpret_cast<TV*>(right), dataVec);
        auto popCount = -_mm_popcnt_u64(mask);
        right += popCount;
        left += popCount + N;
    }

    static INLINE void partition_block_with_compress(TV& dataVec,
                                                     const TV& P,
                                                     T*& left,
                                                     T*& right) {
        auto mask = Tp::get_cmpgt_mask(dataVec, P);
        auto popCount = -_mm_popcnt_u64(mask);
        Tp::store_compress_vec(reinterpret_cast<TV*>(left), dataVec, ~mask);
        Tp::store_compress_vec(reinterpret_cast<TV*>(right + N + popCount), dataVec, mask);
        right += popCount;
        left += popCount + N;
    }

    template<int InnerUnroll>
    T* vectorized_partition(T* const left, T* const right, const AH hint) {
        assert(right - left >= SMALL_SORT_THRESHOLD_ELEMENTS);
        assert((reinterpret_cast<size_t>(left) & ELEMENT_ALIGN) == 0);
        assert((reinterpret_cast<size_t>(right) & ELEMENT_ALIGN) == 0);

        // Vectorized double-pumped (dual-sided) partitioning:
        // We start with picking a pivot using the media-of-3 "method"
        // Once we have sensible pivot stored as the last element of the array
        // We process the array from both ends.
        //
        // To get this rolling, we first read 2 Vector256 elements from the left and
        // another 2 from the right, and store them in some temporary space in order
        // to leave enough "space" inside the vector for storing partitioned values.
        // Why 2 from each side? Because we need n+1 from each side where n is the
        // number of Vector256 elements we process in each iteration... The
        // reasoning behind the +1 is because of the way we decide from *which* side
        // to read, we may end up reading up to one more vector from any given side
        // and writing it in its entirety to the opposite side (this becomes
        // slightly clearer when reading the code below...) Conceptually, the bulk
        // of the processing looks like this after clearing out some initial space
        // as described above:

        // [.............................................................................]
        //  ^wl          ^rl                                               rr^ wr^
        // Where:
        // wl = writeLeft
        // rl = readLeft
        // rr = readRight
        // wr = writeRight

        // In every iteration, we select what side to read from based on how much
        // space is left between head read/write pointer on each side...
        // We read from where there is a smaller gap, e.g. that side
        // that is closer to the unfortunate possibility of its write head
        // overwriting its read head... By reading from THAT side, we're ensuring
        // this does not happen

        // An additional unfortunate complexity we need to deal with is that the
        // right pointer must be decremented by another Vector256<T>.Count elements
        // Since the Load/Store primitives obviously accept start addresses
        auto pivot = *right;
        // We do this here just in case we need to pre-align to the right
        // We end up
        *right = std::numeric_limits<T>::max();

        // Broadcast the selected pivot
        const TV P = Tp::get_vec_pivot(pivot);

        auto readLeft = left;
        auto readRight = right;

        auto tmpStartLeft = _temp;
        auto tmpLeft = tmpStartLeft;
        auto tmpStartRight = _temp + PARTITION_TMP_SIZE_IN_ELEMENTS;
        auto tmpRight = tmpStartRight;

        tmpRight -= N;

        // the read heads always advance by 8 elements, or 32 bytes,
        // We can spend some extra time here to align the pointers
        // so they start at a cache-line boundary
        // Once that happens, we can read with Avx.LoadAlignedVector256
        // And also know for sure that our reads will never cross cache-lines
        // Otherwise, 50% of our AVX2 Loads will need to read from two cache-lines
        align_vectorized(left, right, hint, P, readLeft, readRight,
                         tmpStartLeft, tmpLeft, tmpStartRight, tmpRight);

        const auto leftAlign = hint.left_align;
        const auto rightAlign = hint.right_align;
        if (leftAlign > 0) {
            tmpRight += N;
            readLeft = align_left_scalar_uncommon(readLeft, pivot, tmpLeft, tmpRight);
            tmpRight -= N;
        }

        if (rightAlign < 0) {
            tmpRight += N;
            readRight =
                    align_right_scalar_uncommon(readRight, pivot, tmpLeft, tmpRight);
            tmpRight -= N;
        }

        assert(((size_t)readLeft & ALIGN_MASK) == 0);
        assert(((size_t)readRight & ALIGN_MASK) == 0);

        assert((((size_t)readRight - (size_t)readLeft) % ALIGN) == 0);
        assert((readRight - readLeft) >= InnerUnroll * 2);

        // From now on, we are fully aligned
        // and all reading is done in full vector units
        auto readLeftV = (TV*) readLeft;
        auto readRightV = (TV*) readRight;
        #ifndef NDEBUG
        readLeft = nullptr;
        readRight = nullptr;
        #endif

        for (auto u = 0; u < InnerUnroll; u++) {
            auto dl = Tp::load_vec(readLeftV + u);
            auto dr = Tp::load_vec(readRightV - (u + 1));
            partition_block(dl, P, tmpLeft, tmpRight);
            partition_block(dr, P, tmpLeft, tmpRight);
        }

        tmpRight += N;
        // Adjust for the reading that was made above
        readLeftV  += InnerUnroll;
        readRightV -= InnerUnroll*2;
        TV* nextPtr;

        auto writeLeft = left;
        auto writeRight = right - N;

        while (readLeftV < readRightV) {
            if (writeRight - ((T *) readRightV) < (2 * (InnerUnroll * N) - N)) {
                nextPtr = readRightV;
                readRightV -= InnerUnroll;
            } else {
                mess_up_cmov();
                nextPtr = readLeftV;
                readLeftV += InnerUnroll;
            }

            TV d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12;

            switch (InnerUnroll) {
                case 12: d12 = Tp::load_vec(nextPtr + InnerUnroll - 12);
                case 11: d11 = Tp::load_vec(nextPtr + InnerUnroll - 11);
                case 10: d10 = Tp::load_vec(nextPtr + InnerUnroll - 10);
                case  9: d09 = Tp::load_vec(nextPtr + InnerUnroll -  9);
                case  8: d08 = Tp::load_vec(nextPtr + InnerUnroll -  8);
                case  7: d07 = Tp::load_vec(nextPtr + InnerUnroll -  7);
                case  6: d06 = Tp::load_vec(nextPtr + InnerUnroll -  6);
                case  5: d05 = Tp::load_vec(nextPtr + InnerUnroll -  5);
                case  4: d04 = Tp::load_vec(nextPtr + InnerUnroll -  4);
                case  3: d03 = Tp::load_vec(nextPtr + InnerUnroll -  3);
                case  2: d02 = Tp::load_vec(nextPtr + InnerUnroll -  2);
                case  1: d01 = Tp::load_vec(nextPtr + InnerUnroll -  1);
            }

            switch (InnerUnroll) {
                case 12: partition_block(d12, P, writeLeft, writeRight);
                case 11: partition_block(d11, P, writeLeft, writeRight);
                case 10: partition_block(d10, P, writeLeft, writeRight);
                case  9: partition_block(d09, P, writeLeft, writeRight);
                case  8: partition_block(d08, P, writeLeft, writeRight);
                case  7: partition_block(d07, P, writeLeft, writeRight);
                case  6: partition_block(d06, P, writeLeft, writeRight);
                case  5: partition_block(d05, P, writeLeft, writeRight);
                case  4: partition_block(d04, P, writeLeft, writeRight);
                case  3: partition_block(d03, P, writeLeft, writeRight);
                case  2: partition_block(d02, P, writeLeft, writeRight);
                case  1: partition_block(d01, P, writeLeft, writeRight);
              }
          }

        readRightV += (InnerUnroll - 1);

        while (readLeftV <= readRightV) {
          if (writeRight - (T *) readRightV < N) {
                nextPtr = readRightV;
                readRightV -= 1;
            } else {
                mess_up_cmov();
                nextPtr = readLeftV;
                readLeftV += 1;
            }

            auto d = Tp::load_vec(nextPtr);
            partition_block(d, P, writeLeft, writeRight);
            //partition_block_without_compress(d, P, writeLeft, writeRight);
        }

        // 3. Copy-back the 4 registers + remainder we partitioned in the beginning
        auto leftTmpSize = tmpLeft - tmpStartLeft;
        memcpy(writeLeft, tmpStartLeft, leftTmpSize * sizeof(T));
        writeLeft += leftTmpSize;
        auto rightTmpSize = tmpStartRight - tmpRight;
        memcpy(writeLeft, tmpRight, rightTmpSize * sizeof(T));

        // Shove to pivot back to the boundary
        *right = *writeLeft;
        *writeLeft++ = pivot;

        assert(writeLeft > left);
        assert(writeLeft <= right);

        return writeLeft;
    }
    void align_vectorized(const T* left,
                          const T* right,
                          const AH& hint,
                          const TV P,
                          T*& readLeft,
                          T*& readRight,
                          T*& tmpStartLeft,
                          T*& tmpLeft,
                          T*& tmpStartRight,
                          T*& tmpRight) const {
        const auto leftAlign  = hint.left_align;
        const auto rightAlign = hint.right_align;
        const auto rai = ~((rightAlign - 1) >> 31);
        const auto lai = leftAlign >> 31;
        const auto preAlignedLeft  = (TV*) (left + leftAlign);
        const auto preAlignedRight = (TV*) (right + rightAlign - N);

        // Alignment with vectorization is tricky, so read carefully before changing code:
        // 1. We load data, which we might need to align, if the alignment hints
        //    mean pre-alignment (or overlapping alignment)
        // 2. We partition and store in the following order:
        //    a) right-portion of right vector to the right-side
        //    b) left-portion of left vector to the right side
        //    c) at this point one-half of each partitioned vector has been committed
        //       back to memory.
        //    d) we advance the right write (tmpRight) pointer by how many elements
        //       were actually needed to be written to the right hand side
        //    e) We write the right portion of the left vector to the right side
        //       now that its write position has been updated
        auto RT0 = Tp::load_vec(preAlignedRight);
        auto LT0 = Tp::load_vec(preAlignedLeft);
        auto rtMask = Tp::get_cmpgt_mask(RT0, P);
        auto ltMask = Tp::get_cmpgt_mask(LT0, P);
        const auto rtPopCountRightPart = std::max(_mm_popcnt_u32(rtMask), rightAlign);
        const auto ltPopCountRightPart = _mm_popcnt_u32(ltMask);
        const auto rtPopCountLeftPart  = N - rtPopCountRightPart;
        const auto ltPopCountLeftPart  = N - ltPopCountRightPart;

        if (Tp::supports_compress_writes()) {
          Tp::store_compress_vec((TV *) (tmpRight + N - rtPopCountRightPart), RT0, rtMask);
          Tp::store_compress_vec((TV *) tmpLeft,  LT0, ~ltMask);

          tmpRight -= rtPopCountRightPart & rai;
          readRight += (rightAlign - N) & rai;

          Tp::store_compress_vec((TV *) (tmpRight + N - ltPopCountRightPart), LT0, ltMask);
          tmpRight -= ltPopCountRightPart & lai;
          tmpLeft += ltPopCountLeftPart & lai;
          tmpStartLeft += -leftAlign & lai;
          readLeft += (leftAlign + N) & lai;

          Tp::store_compress_vec((TV*) tmpLeft, RT0, ~rtMask);
          tmpLeft += rtPopCountLeftPart & rai;
          tmpStartRight -= rightAlign & rai;
        }
        else {
            RT0 = Tp::partition_vector(RT0, rtMask);
            LT0 = Tp::partition_vector(LT0, ltMask);
            Tp::store_vec((TV*) tmpRight, RT0);
            Tp::store_vec((TV*) tmpLeft, LT0);


            tmpRight -= rtPopCountRightPart & rai;
            readRight += (rightAlign - N) & rai;

            Tp::store_vec((TV*) tmpRight, LT0);
            tmpRight -= ltPopCountRightPart & lai;

            tmpLeft += ltPopCountLeftPart & lai;
            tmpStartLeft += -leftAlign & lai;
            readLeft += (leftAlign + N) & lai;

            Tp::store_vec((TV*) tmpLeft, RT0);
            tmpLeft += rtPopCountLeftPart & rai;
            tmpStartRight -= rightAlign & rai;
        }
    }

   public:
    NOINLINE void sort(T* left, T* right) {
        init_isa_detection();
        reset(left, right);
        auto depthLimit = 2 * floor_log2_plus_one(right + 1 - left);
        sort(left, right, AH(), depthLimit);
    }
};

}  // namespace gcsort


#include "vxsort_targets_disable.h"

#endif
