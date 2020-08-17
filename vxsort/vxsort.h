#ifndef VXSORT_VXSORT_H
#define VXSORT_VXSORT_H

#ifdef __GNUC__
#ifdef __clang__
#pragma clang attribute push (__attribute__((target("popcnt"))), apply_to = any(function))
#else
#pragma GCC push_options
#pragma GCC target("popcnt")
#endif
#endif


#include <assert.h>
#include <immintrin.h>

#include "defs.h"
#include "isa_detection.h"
#include "alignment.h"
#include "machine_traits.h"
#include "vxsort_stats.h"
#include "packer.h"
#include "smallsort/bitonic_sort.h"

#include <algorithm>
#include <memory>
#include <cstring>
#include <cstdint>
#include <cstdio>

namespace vxsort {

/**
 * sort primitives, quickly
 * @tparam T The primitive type being sorted
 * @tparam M The vector machine model/ISA (e.g. AVX2, AVX512 etc.)
 * @tparam Unroll The unroll factor, controls to some extent, the code-bloat/speedup ration at the call site
 *                Defaults to 1
 * @tparam Shift Optional; specify how many LSB bits are known to be zero in the original input. Can be used
 *               to further speed up sorting.
 */
template <typename T, vector_machine M, int Unroll=1, int Shift=0>
class vxsort {
    static_assert(Unroll >= 1, "Unroll can be in the range 1..12");
    static_assert(Unroll <= 12, "Unroll can be in the range 1..12");

private:
    using MT = vxsort_machine_traits<T, M>;
    typedef typename MT::TPACK TPACK;
    using MT_PACKED = vxsort_machine_traits<TPACK, M>;
    typedef typename MT::TV TV;
    typedef alignment_hint<sizeof(TV)> AH;

    static const int ELEMENT_ALIGN = sizeof(T) - 1;
    static const int N = sizeof(TV) / sizeof(T);
    static const int32_t SMALL_SORT_THRESHOLD_ELEMENTS = 1024; 
    static const int32_t SMALL_SORT_THRESHOLD_VECTORS = SMALL_SORT_THRESHOLD_ELEMENTS / N;
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

    static_assert(PARTITION_TMP_SIZE_IN_ELEMENTS < SMALL_SORT_THRESHOLD_ELEMENTS, "Unroll-level must match small-sorting threshold");
    static const int32_t PackUnroll = (Unroll / 2 > 0) ? Unroll / 2 : 1;


    void reset(T* start, T* end) {
        _depth = 0;
        _startPtr = start;
        _endPtr = end;
    }

    T* _startPtr = nullptr;
    T* _endPtr = nullptr;

    T _temp[PARTITION_TMP_SIZE_IN_ELEMENTS];
    int _depth = 0;

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

    void sort(T* left, T* right, T left_hint, T right_hint, AH realign_hint,
              int depth_limit) {
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
#ifdef VXSORT_STATS
            vxsort_stats<T>::bump_small_sorts();
            vxsort_stats<T>::record_small_sort_size(length);
#endif

            const auto aligned_left = reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(left) & ~(N - 1));
            if (aligned_left < _startPtr) {
                smallsort::bitonic<T, M>::sort(left, length);
                return;
            }

            length += (left - aligned_left);
            smallsort::bitonic<T, M>::sort(aligned_left, length);
            return;
        }

        // Detect a whole bunch of bad cases where partitioning
        // will not do well:
        // 1. Reverse sorted array
        // 2. High degree of repeated values (dutch flag problem, one value)
        if (depth_limit == 0) {
            heap_sort(left, right);
            _depth--;
            return;
        }

        depth_limit--;

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

        if (realign_hint.left_align == AH::REALIGN) {
            // Alignment flow:
            // * Calculate pre-alignment on the left
            // * See it would cause us an out-of bounds read
            // * Since we'd like to avoid that, we adjust for post-alignment
            // * No branches since we do branch->arithmetic
            auto preAlignedLeft = reinterpret_cast<T*>(reinterpret_cast<size_t>(left) & ~ALIGN_MASK);
            auto cannotPreAlignLeft = (preAlignedLeft - _startPtr) >> 63;
            realign_hint.left_align = (preAlignedLeft - left) + (N & cannotPreAlignLeft);
            assert(realign_hint.left_align >= -N && realign_hint.left_align <= N);
            assert(AH::is_aligned(left + realign_hint.left_align));
        }

        if (realign_hint.right_align == AH::REALIGN) {
            // Same as above, but in addition:
            // right is pointing just PAST the last element we intend to partition
            // (it's pointing to where we will store the pivot!) So we calculate alignment based on
            // right - 1
            auto preAlignedRight = reinterpret_cast<T*>(((reinterpret_cast<size_t>(right) - 1) & ~ALIGN_MASK) + ALIGN);
            auto cannotPreAlignRight = (_endPtr - preAlignedRight) >> 63;
            realign_hint.right_align = (preAlignedRight - right - (N & cannotPreAlignRight));
            assert(realign_hint.right_align >= -N && realign_hint.right_align <= N);
            assert(AH::is_aligned(right + realign_hint.right_align));
        }

        // Compute median-of-three, of:
        // the first, mid and one before last elements
        mid = left + ((right - left) / 2);
        swap_if_greater(left, mid);
        swap_if_greater(left, right - 1);
        swap_if_greater(mid, right - 1);

        // Pivot is mid, place it in the right hand side
        swap(mid, right);

        if (MT::supports_packing()) {
            if (MT::template can_pack<Shift>(right_hint - left_hint)) {
                auto left_length = vectorized_packed_partition(left, right, left_hint, realign_hint);
                auto right_length = length - left_length;

                const auto left_packed = reinterpret_cast<TPACK *>(left);
                const auto right_packed = reinterpret_cast<TPACK *>(right);

                auto packed_sorter = vxsort<TPACK, M, Unroll>();
                packed_sorter.sort(left_packed, left_packed + left_length - 1);
                packed_sorter.sort(right_packed + 1 - right_length, right_packed);
                vectorized_unpack_from_edge<Backward, 2>(left_packed, left_length, left_hint);
                vectorized_unpack_from_edge<Forward, 2>(right_packed, right_length, left_hint);
                //packer<T, TPACK, M, Shift, 2, SMALL_SORT_THRESHOLD_ELEMENTS>::unpack((TPACK *) left, length, left_hint);
                return;
            }
        }

        auto sep = vectorized_partition<Unroll>(left, right, realign_hint);

        _depth++;
        sort(left, sep - 2, left_hint, *sep, realign_hint.realign_right(), depth_limit);
        sort(sep, right, *(sep - 2), right_hint, realign_hint.realign_left(), depth_limit);
        _depth--;
    }

    template <typename TFriend, vector_machine MFriend, int UnrollFriend, int ShiftFriend>
    friend class vxsort;

    static INLINE void partition_block(TV& data_vec,
                                       const TV P,
                                       T*& left,
                                       T*& right) {
#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads();
        vxsort_stats<T>::bump_vec_stores(2);
#endif
      if (MT::supports_compress_writes()) {
        partition_block_with_compress(data_vec, P, left, right);
      } else {
        partition_block_without_compress(data_vec, P, left, right);
      }
    }

    static INLINE void partition_block_without_compress(TV& data_vec,
                                                        const TV P,
                                                        T*& left,
                                                        T*& right) {
#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_perms();
#endif
        auto mask = MT::get_cmpgt_mask(data_vec, P);
        data_vec = MT::partition_vector(data_vec, mask);
        MT::store_vec(reinterpret_cast<TV*>(left), data_vec);
        MT::store_vec(reinterpret_cast<TV*>(right), data_vec);
        auto popCount = -_mm_popcnt_u64(mask);
        right += popCount;
        left += popCount + N;
    }

    static INLINE void partition_block_with_compress(TV& data_vec,
                                                     const TV P,
                                                     T*& left,
                                                     T*& right) {
        auto mask = MT::get_cmpgt_mask(data_vec, P);
        auto popCount = -_mm_popcnt_u64(mask);
        MT::store_compress_vec(reinterpret_cast<TV*>(left), data_vec, ~mask);
        MT::store_compress_vec(reinterpret_cast<TV*>(right + N + popCount), data_vec, mask);
        right += popCount;
        left += popCount + N;
    }

    template<int InnerUnroll>
    T* vectorized_partition(T* const left, T* const right, const AH hint) {
        assert(right - left >= SMALL_SORT_THRESHOLD_ELEMENTS);
        assert((reinterpret_cast<size_t>(left) & ELEMENT_ALIGN) == 0);
        assert((reinterpret_cast<size_t>(right) & ELEMENT_ALIGN) == 0);

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_partitions((right - left) + 1);
#endif

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
        const TV P = MT::broadcast(pivot);

        auto readLeft = left;
        auto readRight = right;

        auto tmp_start_left = _temp;
        auto tmp_left = tmp_start_left;
        auto tmp_start_right = _temp + PARTITION_TMP_SIZE_IN_ELEMENTS;
        auto tmp_right = tmp_start_right;

        tmp_right -= N;

        // the read heads always advance by 8 elements, or 32 bytes,
        // We can spend some extra time here to align the pointers
        // so they start at a cache-line boundary
        // Once that happens, we can read with Avx.LoadAlignedVector256
        // And also know for sure that our reads will never cross cache-lines
        // Otherwise, 50% of our AVX2 Loads will need to read from two cache-lines
        align_vectorized(left, right, hint, P, readLeft, readRight, tmp_start_left, tmp_left, tmp_start_right, tmp_right);

        const auto leftAlign = hint.left_align;
        const auto rightAlign = hint.right_align;
        if (leftAlign > 0) {
            tmp_right += N;
            readLeft = align_left_scalar_uncommon(readLeft, pivot, tmp_left, tmp_right);
            tmp_right -= N;
        }

        if (rightAlign < 0) {
            tmp_right += N;
            readRight =
                    align_right_scalar_uncommon(readRight, pivot, tmp_left, tmp_right);
            tmp_right -= N;
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
            auto dl = MT::load_vec(readLeftV + u);
            auto dr = MT::load_vec(readRightV - (u + 1));
            partition_block(dl, P, tmp_left, tmp_right);
            partition_block(dr, P, tmp_left, tmp_right);
        }

        tmp_right += N;
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
                case 12: d12 = MT::load_vec(nextPtr + InnerUnroll - 12);
                case 11: d11 = MT::load_vec(nextPtr + InnerUnroll - 11);
                case 10: d10 = MT::load_vec(nextPtr + InnerUnroll - 10);
                case  9: d09 = MT::load_vec(nextPtr + InnerUnroll -  9);
                case  8: d08 = MT::load_vec(nextPtr + InnerUnroll -  8);
                case  7: d07 = MT::load_vec(nextPtr + InnerUnroll -  7);
                case  6: d06 = MT::load_vec(nextPtr + InnerUnroll -  6);
                case  5: d05 = MT::load_vec(nextPtr + InnerUnroll -  5);
                case  4: d04 = MT::load_vec(nextPtr + InnerUnroll -  4);
                case  3: d03 = MT::load_vec(nextPtr + InnerUnroll -  3);
                case  2: d02 = MT::load_vec(nextPtr + InnerUnroll -  2);
                case  1: d01 = MT::load_vec(nextPtr + InnerUnroll -  1);
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

            auto d = MT::load_vec(nextPtr);
            partition_block(d, P, writeLeft, writeRight);
            //partition_block_without_compress(d, P, writeLeft, writeRight);
        }

        // 3. Copy-back the 4 registers + remainder we partitioned in the beginning
        auto left_tmp_size = tmp_left - tmp_start_left;
        memcpy(writeLeft, tmp_start_left, left_tmp_size * sizeof(T));
        writeLeft += left_tmp_size;
        auto right_tmp_size = tmp_start_right - tmp_right;
        memcpy(writeLeft, tmp_right, right_tmp_size * sizeof(T));

        // Shove to pivot back to the boundary
        *right = *writeLeft;
        *writeLeft++ = pivot;

        assert(writeLeft > left);
        assert(writeLeft <= right+1);

        return writeLeft;
    }

    size_t vectorized_packed_partition(T* const left, T* const right, T base, const AH hint) {
        assert(right - left >= SMALL_SORT_THRESHOLD_ELEMENTS);
        assert((reinterpret_cast<size_t>(left) & ELEMENT_ALIGN) == 0);
        assert((reinterpret_cast<size_t>(right) & ELEMENT_ALIGN) == 0);

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_partitions((right - left) + 1);
#endif
        auto pivot = *right;
        // We do this here just in case we need to pre-align to the right
        // We end up
        *right = std::numeric_limits<T>::max();

        // Broadcast the selected pivot
        T offset = MT::template shift_n_sub<Shift>(base, (T) std::numeric_limits<TPACK>::min());
        const TV P = MT::broadcast(pivot);
        const TV BV = MT::broadcast(offset);

        auto readLeft = left;
        auto readRight = right;

        auto tmp_start_left = _temp;
        auto tmp_left = tmp_start_left;
        auto tmp_start_right = _temp + PARTITION_TMP_SIZE_IN_ELEMENTS;
        auto tmp_right = tmp_start_right;

        tmp_right -= N;

        // the read heads always advance by 8 elements, or 32 bytes,
        // We can spend some extra time here to align the pointers
        // so they start at a cache-line boundary
        // Once that happens, we can read with Avx.LoadAlignedVector256
        // And also know for sure that our reads will never cross cache-lines
        // Otherwise, 50% of our AVX2 Loads will need to read from two cache-lines
        align_vectorized(left, right, hint, P, readLeft, readRight, tmp_start_left, tmp_left, tmp_start_right, tmp_right);

        const auto leftAlign = hint.left_align;
        const auto rightAlign = hint.right_align;
        if (leftAlign > 0) {
            tmp_right += N;
            readLeft = align_left_scalar_uncommon(readLeft, pivot, tmp_left, tmp_right);
            tmp_right -= N;
        }

        if (rightAlign < 0) {
            tmp_right += N;
            readRight =
                align_right_scalar_uncommon(readRight, pivot, tmp_left, tmp_right);
            tmp_right -= N;
        }

        assert(((size_t)readLeft & ALIGN_MASK) == 0);
        assert(((size_t)readRight & ALIGN_MASK) == 0);

        assert((((size_t)readRight - (size_t)readLeft) % ALIGN) == 0);
        //assert((readRight - readLeft) >= InnerUnroll * 2);

        // From now on, we are fully aligned
        // and all reading is done in full vector units
        auto read_left_v = (TV*) readLeft;
        auto read_right_v = (TV*) readRight;
#ifndef NDEBUG
        readLeft = nullptr;
        readRight = nullptr;
#endif

        auto write_left = (TPACK *) left;
        auto write_right = (TPACK *) right - 2*N;

        // We will be packing before partitioning, so
        // We must gnerate a pre-packed pivot
        const auto packed_pivot = MT::template shift_n_sub<Shift>(pivot, offset);
        const TV PPP = MT_PACKED::broadcast(static_cast<TPACK>(packed_pivot));

        auto lenv = read_right_v - read_left_v;
        auto len_dv = lenv / 2;

        lenv -= len_dv * 2;

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads(2 * len_dv);
        vxsort_stats<T>::bump_vec_stores(2 * len_dv);
#endif

        for (auto i = 0; i < len_dv; i++) {
            auto dl = MT::load_vec(read_left_v + i);
            auto dr = MT::load_vec(read_right_v - (i + 1));


            if (Shift > 0) { // This is statically compiled in/out
                dl = MT::shift_right(dl, Shift);
                dr = MT::shift_right(dr, Shift);
            }
            dl = MT::sub(dl, BV);
            dr = MT::sub(dr, BV);

            auto packed_data = MT::pack_unordered(dl, dr);

            vxsort<TPACK, M, Unroll>::partition_block(packed_data, PPP, write_left, write_right);
        }

        // We might have one more vector worth of stuff to partition, so we'll do it with
        // scalar partitioning into the tmp space
        if (lenv > 0) {
            auto slack = MT::load_vec((TV *) (read_left_v + len_dv));
            partition_block(slack, P, tmp_left, tmp_right);
        }

        write_right += 2*N;

        for (auto p = tmp_start_left; p < tmp_left; p++) {
            *(write_left++) = static_cast<TPACK>(MT::template shift_n_sub<Shift>(*p, offset));
        }

        for (auto p = tmp_right; p < tmp_start_right; p++) {
            *(--write_right) = static_cast<TPACK>(MT::template shift_n_sub<Shift>(*p, offset));
        }

        *write_left++ = static_cast<TPACK>(packed_pivot);

        return write_left - ((TPACK *) left);
    }

    enum UnpackDirection {
        Backward = -1,
        Forward = 1,
    };

    static NOINLINE void unpack_vectorized(const TV baseVec, TV d01, TV& u01, TV& u02) {
        MT::unpack_ordered(d01, u01, u02);

        u01 = MT::add(u01, baseVec);
        u02 = MT::add(u02, baseVec);

        if (Shift > 0) { // This is statically compiled in/out
            u01 = MT::shift_left(u01, Shift);
            u02 = MT::shift_left(u02, Shift);
        }
    }


    template<UnpackDirection Dir, int UnpackUnroll>
    void vectorized_unpack_from_edge(TPACK* const mem, size_t len, T base) {
        T offset = MT::template shift_n_sub<Shift>(base, (T) std::numeric_limits<TPACK>::min());
        auto baseVec = MT::broadcast(offset);

        auto mem_read = mem + len;
        auto mem_write = ((T *) mem) + len;

        auto pre_aligned_mem = reinterpret_cast<TPACK *>(reinterpret_cast<size_t>(mem_read) & ~ALIGN_MASK);

        if (pre_aligned_mem < mem_read) {
            len -= (mem_read - pre_aligned_mem);
            while (mem_read > pre_aligned_mem) {
                *(--mem_write) = MT::template unshift_and_add<Shift>(*(--mem_read), offset);
            }
        }

        assert(AH::is_aligned(mem_read));

        auto lenv = len / (N * 2);
        auto memv_read = ((TV *) mem_read) - 1;
        auto memv_write = ((TV *) mem_write) - 2;
        len -= lenv * N * 2;

        while (lenv >= UnpackUnroll) {
            assert(memv_read <= memv_write);

            TV d01, d02, d03, d04;
            TV u01, u02, u03, u04, u05, u06, u07, u08;

            do {
                d01 = MT::load_vec(memv_read + 0);
                if (UnpackUnroll == 1) break;
                d02 = MT::load_vec(memv_read - 1);
                if (UnpackUnroll == 2) break;
                d03 = MT::load_vec(memv_read - 2);
                if (UnpackUnroll == 3) break;
                d04 = MT::load_vec(memv_read - 3);
                break;
            } while(true);

            do {
                unpack_vectorized(baseVec, d01, u01, u02);
                MT::store_vec(memv_write + 0, u01);
                MT::store_vec(memv_write + 1, u02);
                if (UnpackUnroll == 1) break;
                unpack_vectorized(baseVec, d02, u03, u04);
                MT::store_vec(memv_write - 2, u03);
                MT::store_vec(memv_write - 1, u04);
                if (UnpackUnroll == 2) break;
                unpack_vectorized(baseVec, d03, u05, u06);
                MT::store_vec(memv_write - 4, u05);
                MT::store_vec(memv_write - 3, u06);
                if (UnpackUnroll == 3) break;
                unpack_vectorized(baseVec, d04, u07, u08);
                MT::store_vec(memv_write - 6, u07);
                MT::store_vec(memv_write - 5, u08);
                break;
            } while(true);

            memv_read -= UnpackUnroll;
            memv_write -= 2 * UnpackUnroll;
            lenv -= UnpackUnroll;
        }

        if (UnpackUnroll > 1) {
            while (lenv >= 1) {
                assert(memv_read <= memv_write);

                TV d01;
                TV u01, u02;

                d01 = MT::load_vec(memv_read + 0);

                unpack_vectorized(baseVec, d01, u01, u02);
                MT::store_vec(memv_write + 0, u01);
                MT::store_vec(memv_write + 1, u02);

                memv_read--;
                memv_write -= 2;
                lenv--;
            }
        }

        mem_read = (TPACK *) (memv_read + 1);
        mem_write = (T *) (memv_write + 2);

        while (len-- > 0) {
            *(--mem_write) = MT::template unshift_and_add<Shift>(*(--mem_read), offset);
        }
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
        const auto left_align = hint.left_align;
        const auto right_align = hint.right_align;
        const auto rai = ~((right_align - 1) >> 31);
        const auto lai = left_align >> 31;
        const auto pre_aligned_left = (TV*) (left + left_align);
        const auto pre_aligned_right = (TV*) (right + right_align - N);

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_vec_loads(2);
        vxsort_stats<T>::bump_vec_stores(4);
#endif

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
        auto RT0 = MT::load_vec(pre_aligned_right);
        auto LT0 = MT::load_vec(pre_aligned_left);
        auto rtMask = MT::get_cmpgt_mask(RT0, P);
        auto ltMask = MT::get_cmpgt_mask(LT0, P);
        const auto rtPopCountRightPart = std::max(_mm_popcnt_u32(rtMask), right_align);
        const auto ltPopCountRightPart = _mm_popcnt_u32(ltMask);
        const auto rtPopCountLeftPart  = N - rtPopCountRightPart;
        const auto ltPopCountLeftPart  = N - ltPopCountRightPart;

        if (MT::supports_compress_writes()) {
          MT::store_compress_vec((TV *) (tmpRight + N - rtPopCountRightPart), RT0, rtMask);
          MT::store_compress_vec((TV *) tmpLeft,  LT0, ~ltMask);

          tmpRight -= rtPopCountRightPart & rai;
          readRight += (right_align - N) & rai;

          MT::store_compress_vec((TV *) (tmpRight + N - ltPopCountRightPart), LT0, ltMask);
          tmpRight -= ltPopCountRightPart & lai;
          tmpLeft += ltPopCountLeftPart & lai;
          tmpStartLeft += -left_align & lai;
          readLeft += (left_align + N) & lai;

          MT::store_compress_vec((TV*) tmpLeft, RT0, ~rtMask);
          tmpLeft += rtPopCountLeftPart & rai;
          tmpStartRight -= right_align & rai;
        }
        else {
#ifdef VXSORT_STATS
            vxsort_stats<T>::bump_perms(2);
#endif
            RT0 = MT::partition_vector(RT0, rtMask);
            LT0 = MT::partition_vector(LT0, ltMask);
            MT::store_vec((TV*) tmpRight, RT0);
            MT::store_vec((TV*) tmpLeft, LT0);


            tmpRight -= rtPopCountRightPart & rai;
            readRight += (right_align - N) & rai;

            MT::store_vec((TV*) tmpRight, LT0);
            tmpRight -= ltPopCountRightPart & lai;

            tmpLeft += ltPopCountLeftPart & lai;
            tmpStartLeft += -left_align & lai;
            readLeft += (left_align + N) & lai;

            MT::store_vec((TV*) tmpLeft, RT0);
            tmpLeft += rtPopCountLeftPart & rai;
            tmpStartRight -= right_align & rai;
        }
    }

   public:
    /**
     * Sort a given range
     * @param left The left edge of the range, including
     * @param right The right edge of the range, including
     * @param left_hint Optional; A hint, Use to speed up the sorting operation, describing a single value that is known to be
     *        smaller-than, or equalt to all values contained within the provided array.
     * @param right_hint Optional; A hint, Use to speed up the sorting operation, describing a single value that is known to be
     *        larger-than than all values contained within the provided array.
     */
    NOINLINE void sort(T* left, T* right,
                       T left_hint = std::numeric_limits<T>::min(),
                       T right_hint = std::numeric_limits<T>::max())
    {
        init_isa_detection();

#ifdef VXSORT_STATS
        vxsort_stats<T>::bump_sorts((right - left) + 1);
#endif
        reset(left, right);
        auto depthLimit = 2 * floor_log2_plus_one(right + 1 - left);
        sort(left, right, left_hint, right_hint, AH(), depthLimit);
    }
};

}  // namespace gcsort

#include "vxsort_targets_disable.h"

#endif
