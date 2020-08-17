#ifndef VXSORT_MACHINE_TRAITS_AVX2_H
#define VXSORT_MACHINE_TRAITS_AVX2_H

#include "vxsort_targets_enable_avx2.h"

#include <immintrin.h>
#include <stdexcept>
#include <limits>
#include <assert.h>
#include <inttypes.h>
#include "defs.h"
#include "machine_traits.h"

#define i2d _mm256_castsi256_pd
#define d2i _mm256_castpd_si256
#define i2s _mm256_castsi256_ps
#define s2i _mm256_castps_si256
#define s2d _mm256_castps_pd
#define d2s _mm256_castpd_ps



namespace vxsort {

// * We might read the last 4 bytes into a 128-bit vector for 64-bit element masking
// * We might read the last 8 bytes into a 128-bit vector for 32-bit element masking
// This mostly applies to debug mode, since without optimizations, most compilers
// actually execute the instruction stream _mm256_cvtepi8_epiNN + _mm_loadu_si128 as they are given.
// In contrast, release/optimizing compilers, turn that very specific intrinsic pair to
// a more reasonable: vpmovsxbq ymm0, dword [rax*4 + mask_table_4], eliminating the 128-bit
// load completely and effectively reading exactly 4/8 (depending if the instruction is vpmovsxb[q,d]
// without generating an out of bounds read at all.
// But, life is harsh, and we can't trust the compiler to do the right thing if it is not
// contractual, hence this flustercuck
const int M4_SIZE = 16 + 12;
const int M8_SIZE = 64 + 8;

extern const uint8_t mask_table_4[M4_SIZE];
extern const uint8_t mask_table_8[M8_SIZE];

extern const int8_t perm_table_64[128];
extern const int8_t perm_table_32[2048];

template <>
class vxsort_machine_traits<int32_t, AVX2> {
   public:
    typedef int32_t T;
    typedef __m256i TV;
    typedef __m256i TLOADSTOREMASK;
    typedef uint32_t TMASK;
    typedef int32_t TPACK;
    typedef typename std::make_unsigned<T>::type TU;

    static const int N = sizeof(TV) / sizeof(T);

    static constexpr bool supports_compress_writes() { return false; }

    static constexpr bool supports_packing() { return false; }

    template <int Shift>
    static constexpr bool can_pack(T) { return false; }

    static INLINE TLOADSTOREMASK generate_remainder_mask(int remainder) {
        assert(remainder >= 0);
        assert(remainder < 8);
        return _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(mask_table_8 + N * remainder)));
    }

    static INLINE TV load_vec(TV* p) { return _mm256_lddqu_si256(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_si256(ptr, v); }

    static void store_compress_vec(TV*, TV, TMASK) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_masked_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        return _mm256_or_si256(_mm256_maskload_epi32((int32_t *) p, mask),
                               _mm256_andnot_si256(mask, base));
    }

    static INLINE  void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) {
        _mm256_maskstore_epi32((int32_t *) p, mask, v);
    }

    static INLINE TV partition_vector(TV v, int mask) {
        assert(mask >= 0);
        assert(mask <= 255);
        return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(perm_table_32 + mask * 8)))));
    }

    static INLINE TV broadcast(int32_t pivot) { return _mm256_set1_epi32(pivot); }
    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm256_movemask_ps(i2s(_mm256_cmpgt_epi32(a, b))); }

    static TV shift_right(TV v, int i) { return _mm256_srli_epi32(v, i); }
    static TV shift_left(TV v, int i) { return _mm256_slli_epi32(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm256_add_epi32(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm256_sub_epi32(a, b); };

    static INLINE TV pack_ordered(TV , TV ) { TV tmp = _mm256_set1_epi32(0); return tmp; }
    static INLINE TV pack_unordered(TV, TV) { TV tmp = _mm256_set1_epi32(0); return tmp; }
    static INLINE void unpack_ordered(TV, TV&, TV&) { }

    template <int Shift>
    static T shift_n_sub(T v, T sub) {
        if (Shift > 0)
            v >>= Shift;
        v -= sub;
        return v;
    }

    template <int Shift>
    static T unshift_and_add(TPACK from, T add) {
        add += from;
        if (Shift > 0)
            add = (T) (((TU) add) << Shift);
        return add;
    }
};

template <>
class vxsort_machine_traits<uint32_t, AVX2> {
   public:
    typedef uint32_t T;
    typedef __m256i TV;
    typedef __m256i TLOADSTOREMASK;
    typedef uint32_t TMASK;
    typedef uint32_t TPACK;
    typedef typename std::make_unsigned<T>::type TU;

    static const int N = sizeof(TV) / sizeof(T);

    static constexpr bool supports_compress_writes() { return false; }

    static constexpr bool supports_packing() { return false; }

    template <int Shift>
    static constexpr bool can_pack(T) { return false; }

    static INLINE TLOADSTOREMASK generate_remainder_mask(int remainder) {
        assert(remainder >= 0);
        assert(remainder < 8);
        return _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(mask_table_8 + remainder * N)));
    }

    static INLINE TV load_vec(TV* p) { return _mm256_lddqu_si256(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_si256(ptr, v); }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_masked_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        return _mm256_or_si256(_mm256_maskload_epi32((int32_t *) p, mask),
                               _mm256_andnot_si256(mask, base));
    }

    static INLINE  void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) {
        _mm256_maskstore_epi32((int32_t *) p, mask, v);
    }

    static INLINE TV partition_vector(TV v, int mask) {
        assert(mask >= 0);
        assert(mask <= 255);
        return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_32 + mask * 8)))));
    }

    static INLINE TV broadcast(uint32_t pivot) { return _mm256_set1_epi32(pivot); }
    static INLINE TMASK get_cmpgt_mask(TV a, TV b) {
        __m256i top_bit = _mm256_set1_epi32(1U << 31);
        return _mm256_movemask_ps(i2s(_mm256_cmpgt_epi32(_mm256_xor_si256(top_bit, a), _mm256_xor_si256(top_bit, b))));
    }

    static TV shift_right(TV v, int i) { return _mm256_srli_epi32(v, i); }
    static TV shift_left(TV v, int i) { return _mm256_slli_epi32(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm256_add_epi32(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm256_sub_epi32(a, b); };

    static INLINE TV pack_ordered(TV a, TV b) { return a; }
    static INLINE TV pack_unordered(TV a, TV b) { return a; }
    static INLINE void unpack_ordered(TV p, TV& u1, TV& u2) { }

    template <int Shift>
    static T shift_n_sub(T v, T sub) {
        if (Shift > 0)
            v >>= Shift;
        v -= sub;
        return v;
    }

    template <int Shift>
    static T unshift_and_add(TPACK from, T add) {
        add += from;
        if (Shift > 0)
            add = (T) (((TU) add) << Shift);
        return add;
    }
};

template <>
class vxsort_machine_traits<float, AVX2> {
   public:
    typedef float T;
    typedef __m256 TV;
    typedef __m256i TLOADSTOREMASK;
    typedef uint32_t TMASK;
    typedef float TPACK;

    static const int N = sizeof(TV) / sizeof(T);

    static constexpr bool supports_compress_writes() { return false; }

    static constexpr bool supports_packing() { return false; }

    template <int Shift>
    static constexpr bool can_pack(T) { return false; }

    static INLINE TLOADSTOREMASK generate_remainder_mask(int remainder) {
        assert(remainder >= 0);
        assert(remainder < 8);
        return _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(mask_table_8 + remainder * N)));
    }

    static INLINE TV load_vec(TV* p) { return _mm256_loadu_ps((float*)p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_ps((float*)ptr, v); }

    static void store_compress_vec(TV*, TV, TMASK) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_masked_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        return i2s(_mm256_or_si256(s2i(_mm256_maskload_ps((float *) p, mask)),
                               _mm256_andnot_si256(mask, s2i(base))));
    }

    static INLINE  void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) {
        _mm256_maskstore_ps((float *) p, mask, v);
    }

    static INLINE TV partition_vector(TV v, int mask) {
        assert(mask >= 0);
        assert(mask <= 255);
        return _mm256_permutevar8x32_ps(v, _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_32 + mask * 8))));
    }

    static INLINE TV broadcast(float pivot) { return _mm256_set1_ps(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) {
        ///    0x0E: Greater-than (ordered, signaling) \n
        ///    0x1E: Greater-than (ordered, non-signaling)
        return _mm256_movemask_ps(_mm256_cmp_ps(a, b, _CMP_GT_OS));
    }

    static TV shift_right(TV v, int i) { return v; }
    static TV shift_left(TV v, int i) { return v; }

    static INLINE TV add(TV a, TV b) { return _mm256_add_ps(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm256_sub_ps(a, b); };

    static INLINE TV pack_ordered(TV, TV) { throw std::runtime_error("operation is unsupported"); }
    static INLINE TV pack_unordered(TV, TV) { throw std::runtime_error("operation is unsupported"); }
    static INLINE void unpack_ordered(TV, TV&, TV&) { }

    template <int Shift>
    static T shift_n_sub(T v, T sub) { return v; }

    template <int Shift>
    static T unshift_and_add(TPACK from, T add) { return add; }
};

template <>
class vxsort_machine_traits<int64_t, AVX2> {
   public:
    typedef int64_t T;
    typedef __m256i TV;
    typedef __m256i TLOADSTOREMASK;
    typedef uint32_t TMASK;
    typedef int32_t TPACK;
    typedef typename std::make_unsigned<T>::type TU;

    static const int N = sizeof(TV) / sizeof(T);

    static constexpr bool supports_compress_writes() { return false; }

    static constexpr bool supports_packing() { return true; }

    template <int Shift>
    static constexpr bool can_pack(T span) {
        const auto PACK_LIMIT = (((TU) std::numeric_limits<uint32_t>::max() + 1)) << Shift;
        return ((TU) span) < PACK_LIMIT;
    }

    static INLINE TLOADSTOREMASK generate_remainder_mask(int remainder) {
        assert(remainder >= 0);
        assert(remainder < 4);
        return _mm256_cvtepi8_epi64(_mm_loadu_si128((__m128i*)(mask_table_4 + remainder * N)));
    }

    static INLINE TV load_vec(TV* p) { return _mm256_lddqu_si256(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_si256(ptr, v); }

    static void store_compress_vec(TV*, TV, TMASK) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_masked_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        return _mm256_or_si256(_mm256_maskload_epi64((long long *) p, mask),
                               _mm256_andnot_si256(mask, base));
    }

    static INLINE  void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) {
        _mm256_maskstore_epi64((long long *) p, mask, v);
    }

    static INLINE TV partition_vector(TV v, int mask) {
        assert(mask >= 0);
        assert(mask <= 15);
        return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_64 + mask * 8)))));
    }

    static INLINE TV broadcast(int64_t pivot) { return _mm256_set1_epi64x(pivot); }
    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm256_movemask_pd(i2d(_mm256_cmpgt_epi64(a, b))); }

    static TV shift_right(TV v, int i) { return _mm256_srli_epi64(v, i); }
    static TV shift_left(TV v, int i) { return _mm256_slli_epi64(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm256_add_epi64(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm256_sub_epi64(a, b); };

    static INLINE TV pack_ordered(TV a, TV b) {
        a = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(a, _MM_PERM_DBCA), _MM_PERM_DBCA);
        b = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(b, _MM_PERM_DBCA), _MM_PERM_CADB);
        return _mm256_blend_epi32(a, b, 0b11110000);
    }

    static INLINE TV pack_unordered(TV a, TV b) {
        b = _mm256_shuffle_epi32(b, _MM_PERM_CDAB);
        return _mm256_blend_epi32(a, b, 0b10101010);
    }

    static INLINE void unpack_ordered(TV p, TV& u1, TV& u2) {
        auto p01 = _mm256_extracti128_si256(p, 0);
        auto p02 = _mm256_extracti128_si256(p, 1);

        u1 = _mm256_cvtepi32_epi64(p01);
        u2 = _mm256_cvtepi32_epi64(p02);
    }

    template <int Shift>
    static T shift_n_sub(T v, T sub) {
        if (Shift > 0)
            v >>= Shift;
        v -= sub;
        return v;
    }

    template <int Shift>
    static T unshift_and_add(TPACK from, T add) {
        add += from;
        if (Shift > 0)
            add = (T) (((TU) add) << Shift);
        return add;
    }
};

template <>
class vxsort_machine_traits<uint64_t, AVX2> {
   public:
    typedef uint64_t T;
    typedef __m256i TV;
    typedef __m256i TLOADSTOREMASK;
    typedef uint32_t TMASK;
    typedef uint32_t TPACK;
    typedef typename std::make_unsigned<T>::type TU;

    static const int N = sizeof(TV) / sizeof(T);

    static constexpr bool supports_compress_writes() { return false; }

    static constexpr bool supports_packing() { return true; }

    template <int Shift>
    static constexpr bool can_pack(T span) {
        const auto PACK_LIMIT = (((TU) std::numeric_limits<uint32_t>::max() + 1)) << Shift;
        return ((TU) span) < PACK_LIMIT;
    }

    static INLINE TLOADSTOREMASK generate_remainder_mask(int remainder) {
        assert(remainder >= 0);
        assert(remainder < 4);
        return _mm256_cvtepi8_epi64(_mm_loadu_si128((__m128i*)(mask_table_4 + remainder * N)));
    }

    static INLINE TV load_vec(TV* p) { return _mm256_lddqu_si256(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_si256(ptr, v); }

    static void store_compress_vec(TV*, TV, TMASK) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_masked_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        return _mm256_or_si256(_mm256_maskload_epi64((long long *) p, mask),
                               _mm256_andnot_si256(mask, base));
    }

    static INLINE  void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) {
        _mm256_maskstore_epi64((long long *) p, mask, v);
    }

    static INLINE TV partition_vector(TV v, int mask) {
        assert(mask >= 0);
        assert(mask <= 15);
        return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_64 + mask * 8)))));
    }
    static INLINE TV broadcast(int64_t pivot) { return _mm256_set1_epi64x(pivot); }
    static INLINE TMASK get_cmpgt_mask(TV a, TV b) {
        __m256i top_bit = _mm256_set1_epi64x(1LLU << 63);
        return _mm256_movemask_pd(i2d(_mm256_cmpgt_epi64(_mm256_xor_si256(top_bit, a), _mm256_xor_si256(top_bit, b))));
    }

    static INLINE TV shift_right(TV v, int i) { return _mm256_srli_epi64(v, i); }
    static INLINE TV shift_left(TV v, int i) { return _mm256_slli_epi64(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm256_add_epi64(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm256_sub_epi64(a, b); };

    static INLINE TV pack_ordered(TV a, TV b) {
        a = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(a, _MM_PERM_DBCA), _MM_PERM_DBCA);
        b = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(b, _MM_PERM_DBCA), _MM_PERM_CADB);
        return _mm256_blend_epi32(a, b, 0b11110000);
    }

    static INLINE TV pack_unordered(TV a, TV b) {
        b = _mm256_shuffle_epi32(b, _MM_PERM_CDAB);
        return _mm256_blend_epi32(a, b, 0b10101010);
    }

    static INLINE void unpack_ordered(TV p, TV& u1, TV& u2) {
        auto p01 = _mm256_extracti128_si256(p, 0);
        auto p02 = _mm256_extracti128_si256(p, 1);

        u1 = _mm256_cvtepu32_epi64(p01);
        u2 = _mm256_cvtepu32_epi64(p02);
    }

    template <int Shift>
    static T shift_n_sub(T v, T sub) {
        if (Shift > 0)
            v >>= Shift;
        v -= sub;
        return v;
    }

    template <int Shift>
    static T unshift_and_add(TPACK from, T add) {
        add += from;

        if (Shift > 0)
            add = (T) (((TU) add) << Shift);

        return add;
    }
};

template <>
class vxsort_machine_traits<double, AVX2> {
   public:
    typedef double T;
    typedef __m256d TV;
    typedef __m256i TLOADSTOREMASK;
    typedef uint32_t TMASK;
    typedef double TPACK;

    static const int N = sizeof(TV) / sizeof(T);

    static constexpr bool supports_compress_writes() { return false; }

    static constexpr bool supports_packing() { return false; }

    template <int Shift>
    static constexpr bool can_pack(T span) { return false; }

    static INLINE TLOADSTOREMASK generate_remainder_mask(int remainder) {
        assert(remainder >= 0);
        assert(remainder < 4);
        return _mm256_cvtepi8_epi64(_mm_loadu_si128((__m128i*)(mask_table_4 + remainder * N)));
    }

    static INLINE TV load_vec(TV* p) { return _mm256_loadu_pd((double*)p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_pd((double*)ptr, v); }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_masked_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        return i2d(_mm256_or_si256(d2i(_mm256_maskload_pd((double *) p, mask)),
                               _mm256_andnot_si256(mask, d2i(base))));
    }

    static INLINE  void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) {
        _mm256_maskstore_pd((double *) p, mask, v);
    }

    static INLINE TV partition_vector(TV v, int mask) {
        assert(mask >= 0);
        assert(mask <= 15);
        return s2d(_mm256_permutevar8x32_ps(d2s(v), _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_64 + mask * 8)))));
    }

    static INLINE TV broadcast(double pivot) { return _mm256_set1_pd(pivot); }
    static INLINE TMASK get_cmpgt_mask(TV a, TV b) {
        ///    0x0E: Greater-than (ordered, signaling) \n
        ///    0x1E: Greater-than (ordered, non-signaling)
        return _mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_GT_OS));
    }

    static TV shift_right(TV v, int i) { return v; }
    static TV shift_left(TV v, int i) { return v; }

    static INLINE TV add(TV a, TV b) { return _mm256_add_pd(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm256_sub_pd(a, b); };

    static INLINE TV pack_ordered(TV, TV) { TV tmp = _mm256_set1_pd(0); return tmp; }
    static INLINE TV pack_unordered(TV, TV) { TV tmp = _mm256_set1_pd(0); return tmp; }
    static INLINE void unpack_ordered(TV, TV&, TV&) { }

    template <int Shift>
    static T shift_n_sub(T v, T sub) { return v; }

    template <int Shift>
    static T unshift_and_add(TPACK from, T add) { return add; }};

}

#undef i2d
#undef d2i
#undef i2s
#undef s2i
#undef s2d
#undef d2s

#include "vxsort_targets_disable.h"


#endif  // VXSORT_VXSORT_AVX2_H
