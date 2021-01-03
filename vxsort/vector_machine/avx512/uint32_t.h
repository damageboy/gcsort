template <>
class vxsort_machine_traits<uint32_t, AVX512> {
   public:
    typedef uint32_t T;
    typedef __m512i TV;
    typedef __mmask16 TLOADSTOREMASK;
    typedef __mmask16 TMASK;
    typedef uint32_t TPACK;
    typedef typename std::make_unsigned<T>::type TU;

    static const int N = sizeof(TV) / sizeof(T);

    static constexpr bool supports_compress_writes() { return true; }

    static constexpr bool supports_packing() { return false; }

    template <int Shift>
    static constexpr bool can_pack(T span) { return false; }

    static INLINE TLOADSTOREMASK generate_remainder_mask(int remainder) {
        assert(remainder >= 0);
        assert(remainder <= 16);
        return  0xFFFF >> ((N - remainder) & (N-1));
    }

    static INLINE TV load_vec(TV* p) { return _mm512_loadu_si512(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm512_storeu_si512(ptr, v); }

    static TV load_masked_vec(TV *ptr, TV base, TLOADSTOREMASK mask) {
        return _mm512_mask_loadu_epi32(base, mask, (int32_t const *) ptr);
    }

    static INLINE void store_masked_vec(TV * p, TV v, TLOADSTOREMASK mask) {
        _mm512_mask_storeu_epi32(p, mask, v);
    }

    // Will never be called
    static INLINE TV partition_vector(TV v, int mask) { return v; }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) { _mm512_mask_compressstoreu_epi32(ptr, mask, v); }

    static INLINE TV broadcast(uint32_t pivot) { return _mm512_set1_epi32(pivot); }

    static INLINE TMASK get_cmpgt_mask(TV a, TV b) { return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_GT); }

    static TV shift_right(TV v, int i) { return _mm512_srli_epi32(v, i); }
    static TV shift_left(TV v, int i) { return _mm512_slli_epi32(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm512_add_epi32(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm512_sub_epi32(a, b); };

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