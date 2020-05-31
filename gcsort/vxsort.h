#ifndef GCSORT_VXSORT_H
#define GCSORT_VXSORT_H

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include <smallsort/bitonic_sort.AVX2.int64_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.uint64_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.double.generated.h>
#include <smallsort/bitonic_sort.AVX2.float.generated.h>
#include <smallsort/bitonic_sort.AVX2.int32_t.generated.h>
#include <smallsort/bitonic_sort.AVX2.uint32_t.generated.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

#if _MSC_VER
#ifdef _M_X86
#define ARCH_X86
#endif
#ifdef _M_X64
#define ARCH_X64
#endif
#ifdef _M_ARM64
#define ARCH_ARM
#endif
#else
#ifdef __i386__
#define ARCH_X86
#endif
#ifdef __amd64__
#define ARCH_X64
#endif
#ifdef __arm__
#define ARCH_ARM
#endif
#endif

#ifdef _MSC_VER
// MSVC
#include <intrin.h>
#define mess_up_cmov() _ReadBarrier();
#define INLINE __forceinline
#define NOINLINE __declspec(noinline)
#else
// GCC + Clang
#define mess_up_cmov()
#define INLINE __attribute__((always_inline))
#define NOINLINE __attribute__((noinline))
#endif

#define i2d _mm256_castsi256_pd
#define d2i _mm256_castpd_si256
#define i2s _mm256_castsi256_ps
#define s2i _mm256_castps_si256
#define s2d _mm256_castps_pd
#define d2s _mm256_castpd_ps


namespace gcsort {
using gcsort::smallsort::bitonic;

struct alignment_hint {
    public:
        static const size_t ALIGN = 32;
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

enum vector_machine {
  AVX2,
  AVX512,
  SVE,
};

template <typename T, vector_machine M>
struct vxsort_machine_traits {
public:
    typedef __m256 Tv;

    static Tv load_vec(Tv* ptr);
    static Tv store_vec(Tv* ptr, Tv v);
    //static __m256i get_perm(int mask);
    static Tv partition_vector(Tv v, int mask);
    static Tv get_vec_pivot(T pivot);
    static uint32_t get_cmpgt_mask(Tv a, Tv b);
};

#ifdef ARCH_X64

extern const int8_t perm_table_64[128];
extern const int8_t perm_table_32[2048];

template <>
class vxsort_machine_traits<int64_t, AVX2> {
private:
public:
    typedef __m256i Tv;

    static INLINE Tv load_vec(Tv* p) {
        return _mm256_lddqu_si256(p);
    }

    static INLINE void store_vec(Tv* ptr, Tv v) {
      _mm256_storeu_si256(ptr, v);
    }

    static INLINE Tv partition_vector(Tv v, int mask) {
        assert(mask >= 0);
        assert(mask <= 15);
        return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_64 + mask * 8)))));
    }

    static INLINE  Tv get_vec_pivot(int64_t pivot) {
        return _mm256_set1_epi64x(pivot);
    }
    static INLINE uint32_t get_cmpgt_mask(Tv a, Tv b) {
        return _mm256_movemask_pd(i2d(_mm256_cmpgt_epi64(a, b)));
    }
};

template <>
class vxsort_machine_traits<uint64_t, AVX2> {
 private:
 public:
  typedef __m256i Tv;

  static INLINE Tv load_vec(Tv* p) {
    return _mm256_lddqu_si256(p);
  }

  static INLINE void store_vec(Tv* ptr, Tv v) {
    _mm256_storeu_si256(ptr, v);
  }

  static INLINE Tv partition_vector(Tv v, int mask) {
    assert(mask >= 0);
    assert(mask <= 15);
    return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_64 + mask * 8)))));
  }
  static INLINE  Tv get_vec_pivot(int64_t pivot) {
    return _mm256_set1_epi64x(pivot);
  }
  static INLINE uint32_t get_cmpgt_mask(Tv a, Tv b) {
    __m256i top_bit = _mm256_set1_epi64x(1LLU << 63);
    return _mm256_movemask_pd(i2d(_mm256_cmpgt_epi64(_mm256_xor_si256(top_bit, a), _mm256_xor_si256(top_bit, b))));
  }
};

template <>
class vxsort_machine_traits<double, AVX2> {
 private:
 public:
  typedef __m256d Tv;

  static INLINE Tv load_vec(Tv* p) {
    return _mm256_loadu_pd((double *) p);
  }

  static INLINE void store_vec(Tv* ptr, Tv v) {
    _mm256_storeu_pd((double *) ptr, v);
  }

  static INLINE Tv partition_vector(Tv v, int mask) {
    assert(mask >= 0);
    assert(mask <= 15);
    return s2d(_mm256_permutevar8x32_ps(d2s(v), _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_64 + mask * 8)))));
  }

  static INLINE  Tv get_vec_pivot(double pivot) {
    return _mm256_set1_pd(pivot);
  }
  static INLINE uint32_t get_cmpgt_mask(Tv a, Tv b) {
    ///    0x0E: Greater-than (ordered, signaling) \n
    ///    0x1E: Greater-than (ordered, non-signaling)
    return _mm256_movemask_pd(_mm256_cmp_pd(a, b, 0x0E));
  }
};

template <>
class vxsort_machine_traits<int32_t, AVX2> {
public:
    typedef __m256i Tv;
    static INLINE Tv load_vec(Tv* p) {
        return _mm256_lddqu_si256(p);
    }

    static INLINE void store_vec(Tv* ptr, Tv v) {
      _mm256_storeu_si256(ptr, v);
    }

    static INLINE Tv partition_vector(Tv v, int mask) {
      assert(mask >= 0);
      assert(mask <= 255);
      return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_32 + mask * 8)))));
    }

    static INLINE Tv get_vec_pivot(int32_t pivot) {
        return _mm256_set1_epi32(pivot);
    }
    static INLINE  uint32_t get_cmpgt_mask(Tv a, Tv b) {
        return _mm256_movemask_ps(i2s(_mm256_cmpgt_epi32(a, b)));
    }
};

template <>
class vxsort_machine_traits<uint32_t, AVX2> {
 public:
  typedef __m256i Tv;
  static INLINE Tv load_vec(Tv* p) {
    return _mm256_lddqu_si256(p);
  }

  static INLINE void store_vec(Tv* ptr, Tv v) {
    _mm256_storeu_si256(ptr, v);
  }

  static INLINE Tv partition_vector(Tv v, int mask) {
    assert(mask >= 0);
    assert(mask <= 255);
    return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_32 + mask * 8)))));
  }

  static INLINE Tv get_vec_pivot(uint32_t pivot) {
    return _mm256_set1_epi32(pivot);
  }
  static INLINE  uint32_t get_cmpgt_mask(Tv a, Tv b) {
    __m256i top_bit = _mm256_set1_epi32(1U << 31);
    return _mm256_movemask_ps(i2s(_mm256_cmpgt_epi32(_mm256_xor_si256(top_bit, a), _mm256_xor_si256(top_bit, b))));
  }
};

template <>
class vxsort_machine_traits<float, AVX2> {
 public:
  typedef __m256 Tv;
  static INLINE Tv load_vec(Tv* p) {
    return _mm256_loadu_ps((float *)p);
  }

  static INLINE void store_vec(Tv* ptr, Tv v) {
    _mm256_storeu_ps((float *) ptr, v);
  }

  static INLINE Tv partition_vector(Tv v, int mask) {
    assert(mask >= 0);
    assert(mask <= 255);
    return _mm256_permutevar8x32_ps(v, _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_32 + mask * 8))));
  }

  static INLINE Tv get_vec_pivot(float pivot) {
    return _mm256_set1_ps(pivot);
  }

  static INLINE  uint32_t get_cmpgt_mask(Tv a, Tv b) {
    ///    0x0E: Greater-than (ordered, signaling) \n
    ///    0x1E: Greater-than (ordered, non-signaling)
    return _mm256_movemask_ps(_mm256_cmp_ps(a, b, 0x0E));
  }
};

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


template <typename T, vector_machine M, int Unroll=1>
class vxsort {
    static_assert(Unroll >= 1, "Unroll can be in the range 1..12");
    static_assert(Unroll <= 12, "Unroll can be in the range 1..12");

private:
    //using Tv2 = Tp::Tv;
    using Tp = vxsort_machine_traits<T, M>;
    typedef typename Tp::Tv TV;

    static const int ELEMENT_ALIGN = sizeof(T) - 1;
    static const int N = 32 / sizeof(T);
    static const int32_t MAX_BITONIC_SORT_VECTORS = 16;
    static const int32_t SMALL_SORT_THRESHOLD_ELEMENTS = MAX_BITONIC_SORT_VECTORS * N;
    //static const int32_t MaxInnerUnroll = ((SMALL_SORT_THRESHOLD_ELEMENTS - (N - 2*N)) / (2 * N));
    static const int32_t MaxInnerUnroll = (MAX_BITONIC_SORT_VECTORS - 3) / 2;
    static const int32_t SafeInnerUnroll = MaxInnerUnroll > Unroll ? Unroll : MaxInnerUnroll;
    static const int32_t SLACK_PER_SIDE_IN_VECTORS = Unroll;
    static const size_t ALIGN = alignment_hint::ALIGN;
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

    void sort(T* left, T* right,
              alignment_hint realignHint,
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
                bitonic<T>::sort(fakeLeft, nextLength);
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

        if (realignHint.left_align == alignment_hint::REALIGN) {
            // Alignment flow:
            // * Calculate pre-alignment on the left
            // * See it would cause us an out-of bounds read
            // * Since we'd like to avoid that, we adjust for post-alignment
            // * No branches since we do branch->arithmetic
            auto preAlignedLeft = reinterpret_cast<T*>(reinterpret_cast<size_t>(left) & ~ALIGN_MASK);
            auto cannotPreAlignLeft = (preAlignedLeft - _startPtr) >> 63;
            realignHint.left_align = (preAlignedLeft - left) + (N & cannotPreAlignLeft);
            assert(alignment_hint::is_aligned(left + realignHint.left_align));
        }

        if (realignHint.right_align == alignment_hint::REALIGN) {
            // Same as above, but in addition:
            // right is pointing just PAST the last element we intend to partition
            // (it's pointing to where we will store the pivot!) So we calculate alignment based on
            // right - 1
            auto preAlignedRight = reinterpret_cast<T*>(((reinterpret_cast<size_t>(right) - 1) & ~ALIGN_MASK) + ALIGN);
            auto cannotPreAlignRight = (_endPtr - preAlignedRight) >> 63;
            realignHint.right_align = (preAlignedRight - right - (N & cannotPreAlignRight));
            assert(alignment_hint::is_aligned(right + realignHint.right_align));
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
        auto mask = Tp::get_cmpgt_mask(dataVec, P);
        dataVec = Tp::partition_vector(dataVec, mask);
        Tp::store_vec(reinterpret_cast<TV*>(left), dataVec);
        Tp::store_vec(reinterpret_cast<TV*>(right), dataVec);
        auto popCount = -_mm_popcnt_u64(mask);
        right += popCount;
        left += popCount + N;
    }

    template<int InnerUnroll>
    T* vectorized_partition(T* const left, T* const right, const alignment_hint hint) {
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

        const auto leftAlign = hint.left_align;
        const auto rightAlign = hint.right_align;

        auto preAlignedLeft = (TV*) (left + leftAlign);
        auto preAlignedRight = (TV*) (right + rightAlign - N);

        // Read overlapped data from right (includes re-reading the pivot)
        auto RT0 = Tp::load_vec(preAlignedRight);
        auto LT0 = Tp::load_vec(preAlignedLeft);
        auto rtMask = Tp::get_cmpgt_mask(RT0, P);
        auto ltMask = Tp::get_cmpgt_mask(LT0, P);
        auto rtPopCount = std::max(_mm_popcnt_u32(rtMask), rightAlign);
        auto ltPopCount = _mm_popcnt_u32(ltMask);
        RT0 = Tp::partition_vector(RT0, rtMask);
        LT0 = Tp::partition_vector(LT0, ltMask);
        Tp::store_vec((TV*) tmpRight, RT0);
        Tp::store_vec((TV*) tmpLeft, LT0);

        auto rai = ~((rightAlign - 1) >> 31);
        auto lai = leftAlign >> 31;

        tmpRight -= rtPopCount & rai;
        rtPopCount = N - rtPopCount;
        readRight += (rightAlign - N) & rai;

        Tp::store_vec((TV*) tmpRight, LT0);
        tmpRight -= ltPopCount & lai;
        ltPopCount = N - ltPopCount;
        tmpLeft += ltPopCount & lai;
        tmpStartLeft += -leftAlign & lai;
        readLeft += (leftAlign + N) & lai;

        Tp::store_vec((TV*) tmpLeft, RT0);
        tmpLeft += rtPopCount & rai;
        tmpStartRight -= rightAlign & rai;

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
public:
    NOINLINE void sort(T* left, T* right) {
        reset(left, right);
        auto depthLimit = 2 * floor_log2_plus_one(right + 1 - left);
        sort(left, right, alignment_hint(), depthLimit);
    }
};

}  // namespace gcsort
#endif
