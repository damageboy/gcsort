#ifndef VXSORT_MACHINE_TRAITS_H
#define VXSORT_MACHINE_TRAITS_H

#include <cstdint>

namespace vxsort {

enum vector_machine {
    NONE,
    AVX2,
    AVX512,
    SVE,
};

template <typename T, vector_machine M>
struct vxsort_machine_traits {
   public:
    typedef T TV;
    typedef T TLOADSTOREMASK;
    typedef T TMASK;
    typedef T TPACK;

    static constexpr bool supports_compress_writes() {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
        return false;
    }

    static constexpr bool supports_packing() {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
        return false;
    }

    template <int Shift>
    static constexpr bool can_pack(T span) {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
        return false;
    }

    static TV load_vec(TV* ptr) {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
    }

    static TLOADSTOREMASK generate_remainder_mask(int remainder) {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
    }

    static void store_vec(TV* ptr, TV v) {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
    }

    static TV load_masked_vec(TV *ptr, TV base, TLOADSTOREMASK mask) {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
    }

    static void store_masked_vec(TV *ptr, TV v, TLOADSTOREMASK mask) {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
    }

    static void store_compress_vec(TV* ptr, TV v, TMASK mask) {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
    }
    static TV partition_vector(TV v, int mask);
    static TV broadcast(T pivot);
    static TMASK get_cmpgt_mask(TV a, TV b);

    static TV shift_right(TV v, int i);
    static TV shift_left(TV v, int i);

    static TV add(TV a, TV b);
    static TV sub(TV a, TV b);

    static TV pack_ordered(TV a, TV b);
    static TV pack_unordered(TV a, TV b);

    static void unpack_ordered(TV p, TV& u1, TV& u2) {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
    }

    template <int Shift>
    static T shift_n_sub(T v, T sub) {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
        return v;
    }

    template <int Shift>
    static T unshift_and_add(TPACK from, T add) {
        static_assert(sizeof(TV) != sizeof(TV), "func must be specialized!");
        return add;
    }
};

}



#endif  // VXSORT_MACHINE_TRAITS_H
