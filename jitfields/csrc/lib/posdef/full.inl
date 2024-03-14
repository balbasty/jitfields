#ifndef JF_POSDEF_FULL
#define JF_POSDEF_FULL

template <typename offset_t, int C>
struct utils<type::Full, offset_t, C>: public common_sym<offset_t, C>
{
    static constexpr bool need_buffer = true;
    static constexpr offset_t work_size = C*C;

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    matvec(optr_t o, hptr_t h, iptr_t i,
           reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c, ++o) {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t cc = 0; cc < C; ++cc)
                internal::iaddcmul<reduce_t>(acc, h[c*C+cc], i[cc]);
            internal::set(*o, acc);
        }
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    addmatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c, ++o) {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t cc = 0; cc < C; ++cc)
                internal::iaddcmul<reduce_t>(acc, h[c*C+cc], i[cc]);
            internal::iadd<reduce_t>(*o, acc);
        }
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c, ++o) {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t cc = 0; cc < C; ++cc)
                internal::iaddcmul<reduce_t>(acc, h[c*C+cc], i[cc]);
            internal::isub<reduce_t>(*o, acc);
        }
    }

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_(vptr_t v, hptr_t h,
                wptr_t w = nullptr, bptr_t b = nullptr,
                reduce_t unused = static_cast<reduce_t>(0))
    {
        copy_((C*(C+1))/2, b, h);
        for (offset_t c = 0; c < C; ++c)
            internal::iadd<reduce_t>(b[c*C+c], w[c]);
        cholesky<offset_t, C>::decompose_(b, unused);  // cholesky decomposition
        cholesky<offset_t, C>::solve_(b, v, unused);   // solve linear system inplace
    }
};


template <typename offset_t>
struct utils<type::Full, offset_t, 0>: public common_sym<offset_t, 0>
{
    static constexpr bool need_buffer = true;

    static inline offset_t work_size(offset_t C) { return C*C; }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    matvec(offset_t C, optr_t o, hptr_t h, iptr_t i,
           reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c, ++o) {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t cc = 0; cc < C; ++cc)
                internal::iaddcmul<reduce_t>(acc, h[c*C+cc], i[cc]);
            internal::set(*o, acc);
        }
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    addmatvec_(offset_t C, optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c, ++o) {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t cc = 0; cc < C; ++cc)
                internal::iaddcmul<reduce_t>(acc, h[c*C+cc], i[cc]);
            internal::iadd<reduce_t>(*o, acc);
        }
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(offset_t C, optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c, ++o) {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t cc = 0; cc < C; ++cc)
                internal::iaddcmul<reduce_t>(acc, h[c*C+cc], i[cc]);
            internal::isub<reduce_t>(*o, acc);
        }
    }

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_(offset_t C, vptr_t v, hptr_t h,
                wptr_t w = nullptr, bptr_t b = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        copy_((C*(C+1))/2, b, h);
        for (offset_t c = 0; c < C; ++c)
            internal::iadd<reduce_t>(b[c*C+c], w[c]);
        cholesky<offset_t>::decompose_(C, b, static_cast<reduce_t>(0));  // cholesky decomposition
        cholesky<offset_t>::solve_(C, b, v, static_cast<reduce_t>(0));   // solve linear system inplace
    }
};

#endif // JF_POSDEF_FULL
