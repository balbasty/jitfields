#ifndef JF_POSDEF_EYE
#define JF_POSDEF_EYE

template <typename offset_t, int C>
struct utils<type::Eye, offset_t, C>: public common_eye<offset_t, C>
{
    template <typename optr_t, typename iptr_t, typename hptr_t,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    addmatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t hh = static_cast<reduce_t>(h[0]);
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
            internal::iadd<reduce_t>(o[c], hh, i[c]);
    }

    template <typename optr_t, typename iptr_t, typename hptr_t,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t hh = static_cast<reduce_t>(h[0]);
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
            internal::isubcmul<reduce_t>(o[c], hh, i[c]);
    }

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<
                                  vptr_t, hptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_(vptr_t v, hptr_t h,
                wptr_t w = nullptr, bptr_t /*b*/ = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        if (w)
        {
            reduce_t hh = static_cast<reduce_t>(h[0]);
#           pragma unroll
            for (offset_t c = 0; c < C; ++c)
                internal::idivcadd<reduce_t>(v[c], hh, w[c]);
        }
        else
        {
            reduce_t hh = static_cast<reduce_t>(h[0]);
#           pragma unroll
            for (offset_t c = 0; c < C; ++c)
                internal::idiv<reduce_t>(v[c], hh);
        }
    }

    template <typename optr_t, typename hptr_t,
              typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, bptr_t>::value>
    static inline __device__
    void invert(optr_t o, hptr_t h, bptr_t /*b*/ = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        internal::div<reduce_t>(*o, 1., *h);
    }
};

template <typename offset_t>
struct utils<type::Eye, offset_t, 0>: public common_eye<offset_t, 0>
{
    template <typename optr_t, typename iptr_t, typename hptr_t,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    addmatvec_(offset_t C, optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t hh = static_cast<reduce_t>(h[0]);
        for (offset_t c = 0; c < C; ++c)
            internal::iadd<reduce_t>(o[c], hh, i[c]);
    }

    template <typename optr_t, typename iptr_t, typename hptr_t,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(offset_t C, optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t hh = static_cast<reduce_t>(h[0]);
        for (offset_t c = 0; c < C; ++c)
            internal::isubcmul<reduce_t>(o[c], hh, i[c]);
    }

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<
                                  vptr_t, hptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_(offset_t C, vptr_t v, hptr_t h,
                wptr_t w = nullptr, bptr_t /*b*/ = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        if (w)
        {
            reduce_t hh = static_cast<reduce_t>(h[0]);
            for (offset_t c = 0; c < C; ++c)
                internal::idivcadd<reduce_t>(v[c], hh, w[c]);
        }
        else
        {
            reduce_t hh = static_cast<reduce_t>(h[0]);
            for (offset_t c = 0; c < C; ++c)
                internal::idiv<reduce_t>(v[c], hh);
        }
    }

    template <typename optr_t, typename hptr_t,
              typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, bptr_t>::value>
    static inline __device__
    void invert(offset_t C, optr_t o, hptr_t h, bptr_t /*b*/ = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        internal::div<reduce_t>(*o, 1., *h);
    }
};

#endif // JF_POSDEF_EYE
