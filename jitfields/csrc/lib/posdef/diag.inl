#ifndef JF_POSDEF_DIAG
#define JF_POSDEF_DIAG


template <typename offset_t, int C>
struct utils<type::Diag, offset_t, C>: public common_diag<offset_t, C>
{
    template <
        typename optr_t,
        typename iptr_t,
        typename hptr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__ void
    matvec(
        optr_t o,
        hptr_t h,
        iptr_t i,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
            internal::mul<reduce_t>(o[c], h[c], i[c]);
    }

    template <
        typename optr_t,
        typename iptr_t,
        typename hptr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__ void
    addmatvec_(
        optr_t o,
        hptr_t h,
        iptr_t i,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
            internal::iaddcmul<reduce_t>(o[c], h[c], i[c]);
    }

    template <
        typename optr_t,
        typename iptr_t,
        typename hptr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__ void
    submatvec_(
        optr_t o,
        hptr_t h,
        iptr_t i,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
            internal::isubcmul<reduce_t>(o[c], h[c], i[c]);
    }

    template <
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value
        >
    static inline __device__ void
    solve_impl_(
        vptr_t v,
        hptr_t h,
        wptr_t w = nullptr,
        bptr_t /*b*/ = nullptr,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        if (w)
        {
#           pragma unroll
            for (offset_t c = 0; c < C; ++c)
                internal::idivcadd<reduce_t>(v[c], h[c], w[c]);
        }
        else
        {
#           pragma unroll
            for (offset_t c = 0; c < C; ++c)
                internal::idiv<reduce_t>(v[c], h[c]);
        }
    }

    template <
        typename optr_t,
        typename hptr_t,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, bptr_t>::value
        >
    static inline __device__
    void invert(
        optr_t o,
        hptr_t h,
        bptr_t /*b*/ = nullptr,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
            internal::div<reduce_t>(o[c], 1., h[c]);
    }
};


template <typename offset_t>
struct utils<type::Diag, offset_t, 0>: public common_diag<offset_t, 0>
{
    template <
        typename optr_t,
        typename iptr_t,
        typename hptr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__
    void matvec(
        offset_t C,
        optr_t o,
        hptr_t h,
        iptr_t i,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c)
            internal::mul<reduce_t>(o[c], h[c], i[c]);
    }

    template <
        typename optr_t,
        typename iptr_t,
        typename hptr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__
    void addmatvec_(
        offset_t C,
        optr_t o,
        hptr_t h,
        iptr_t i,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c)
            internal::iaddcmul<reduce_t>(o[c], h[c], i[c]);
    }

    template <
        typename optr_t,
        typename iptr_t,
        typename hptr_t,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, iptr_t>::value
        >
    static inline __device__
    void submatvec_(
        offset_t C,
        optr_t o,
        hptr_t h,
        iptr_t i,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c)
            internal::isubcmul<reduce_t>(o[c], h[c], i[c]);
    }

    template <
        typename vptr_t,
        typename hptr_t,
        typename wptr_t = const void *,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value
        >
    static inline __device__
    void solve_impl_(
        offset_t C,
        vptr_t v,
        hptr_t h,
        wptr_t w = nullptr,
        bptr_t /*b*/ = nullptr,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        if (w)
        {
            for (offset_t c = 0; c < C; ++c)
                internal::idivcadd<reduce_t>(v[c], h[c], w[c]);
        }
        else
        {
            for (offset_t c = 0; c < C; ++c)
                internal::idiv<reduce_t>(v[c], h[c]);
        }
    }

    template <
        typename optr_t,
        typename hptr_t,
        typename bptr_t = const void *,
        typename reduce_t = typename
            internal::return_type<optr_t, hptr_t, bptr_t>::value
        >
    static inline __device__
    void invert(
        offset_t C,
        optr_t o,
        hptr_t h,
        bptr_t /*b*/ = nullptr,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c)
            internal::div<reduce_t>(o[c], 1., h[c]);
    }
};

#endif // JF_POSDEF_DIAG
