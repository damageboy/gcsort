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
        assert(remainder < N);
        return _mm256_cvtepi8_epi64(_mm_loadu_si128((__m128i*)(mask_table_4 + N * remainder)));
    }

    static INLINE TV load_vec(TV* p) { return _mm256_lddqu_si256(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_si256(ptr, v); }

    static void store_compress_vec(TV*, TV, TMASK) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_masked_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        return _mm256_or_si256(_mm256_maskload_epi64((int64_t *) p, mask),
                               _mm256_andnot_si256(mask, base));
    }

    static INLINE  void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) {
        _mm256_maskstore_epi64((int64_t *) p, mask, v);
    }

    static INLINE TV partition_vector(TV v, int mask) {
        assert(mask >= 0);
        assert(mask <= 15);
        return s2i(_mm256_permutevar8x32_ps(i2s(v), _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(perm_table_64 + mask * 8)))));
    }

    static INLINE TV broadcast(int64_t pivot) { return _mm256_set1_epi64x(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm256_movemask_pd(i2d(_mm256_cmpgt_epi64(a, b))); }

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