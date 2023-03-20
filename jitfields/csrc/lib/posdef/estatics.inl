#ifndef JF_POSDEF_ESTATICS
#define JF_POSDEF_ESTATICS

template <typename offset_t, int C>
struct utils<type::ESTATICS, offset_t, C>: public common_estatics<offset_t, C>
{
    template <typename optr_t, typename iptr_t, typename hptr_t,
              typename reduce_t = typename internal::return_type<optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    matvec(optr_t o, hptr_t h, iptr_t i,
           reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        auto hh = h + C;         // pointer to off-diagonal elements
        reduce_t ii = static_cast<reduce_t>(i[C-1]);    // last input element
        reduce_t oo = static_cast<reduce_t>(0);
        reduce_t hc, hhc, ic;
#       pragma unroll
        for (offset_t c = 0; c < C-1; ++c, ++h, ++hh, ++i, ++o) {
            internal::set(hhc, *hh);
            internal::set(hc, *h);
            internal::set(ic, *i);
            internal::set(*o, hc * ic + hhc * ii);
            internal::iaddcmul<reduce_t>(oo, hhc, ic);
        }
        internal::iaddcmul<reduce_t>(oo, *h, ii);
        internal::set(*o, oo);
    }

    template <typename optr_t, typename iptr_t, typename hptr_t,
              typename reduce_t = typename internal::return_type<optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    addmatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        auto hh = h + C;         // pointer to off-diagonal elements
        reduce_t ii = static_cast<reduce_t>(i[C-1]);    // last input element
        reduce_t oo = static_cast<reduce_t>(o[C-1]);    // last output element
        reduce_t hc, hhc, ic;
#       pragma unroll
        for (offset_t c = 0; c < C-1; ++c, ++h, ++hh, ++i, ++o) {
            internal::set(hhc, *hh);
            internal::set(hc, *h);
            internal::set(ic, *i);
            internal::iadd<reduce_t>(*o, hc * ic + hhc * ii);
            internal::iaddcmul<reduce_t>(oo, hhc, ic);
        }
        internal::iaddcmul<reduce_t>(oo, *h, ii);
        internal::set(*o, oo);
    }

    template <typename optr_t, typename iptr_t, typename hptr_t,
              typename reduce_t = typename internal::return_type<optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        auto hh = h + C;         // pointer to off-diagonal elements
        reduce_t ii = static_cast<reduce_t>(i[C-1]);    // last input element
        reduce_t oo = static_cast<reduce_t>(o[C-1]);    // last output element
        reduce_t hc, hhc, ic;
#       pragma unroll
        for (offset_t c = 0; c < C-1; ++c, ++h, ++hh, ++i, ++o) {
            internal::set(hhc, *hh);
            internal::set(hc, *h);
            internal::set(ic, *i);
            internal::isub<reduce_t>(*o, hc * ic + hhc * ii);
            internal::isubcmul<reduce_t>(oo, hhc, ic);
        }
        internal::isubcmul<reduce_t>(oo, *h, ii);
        internal::set(*o, oo);
    }

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_(vptr_t v, hptr_t h,
                wptr_t w = nullptr, bptr_t /*b*/ = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        auto hh = h + C;         // pointer to off-diagonal elements
        reduce_t ov = static_cast<reduce_t>(0), oh;
        internal::add<reduce_t>(oh, h[C-1], w[C-1]);
#       pragma unroll
        for (offset_t c = 0; c < C-1; ++c, ++h, ++hh, ++v, ++w) {
            reduce_t hc = static_cast<reduce_t>(*h);
            reduce_t hhc = static_cast<reduce_t>(*hh);
            reduce_t tmp = hhc / hc;
            internal::iadd<reduce_t>(hc, *w);
            internal::isubcmul<reduce_t>(oh, hhc, tmp);
            internal::iaddcmul<reduce_t>(ov, *v, tmp);
        }
        oh = 1. / oh; // oh = 1/mini_inv, ov = sum(vec_norm * grad)
        reduce_t tmp = (static_cast<reduce_t>(*v) - ov) * oh;
        internal::set(*v, tmp);
        --v; --h; --hh;
#       pragma unroll
        for (offset_t c = 0; c < C-1; ++c, --v, --h, --hh) {
            reduce_t vc = static_cast<reduce_t>(*v);
            reduce_t hc = static_cast<reduce_t>(*h);
            reduce_t hhc = static_cast<reduce_t>(*hh);
            internal::set(*v, (vc - tmp * hhc) / hc);
        }
    }
};


template <typename offset_t>
struct utils<type::ESTATICS, offset_t, 0>: public common_estatics<offset_t, 0>
{
    template <typename optr_t, typename iptr_t, typename hptr_t,
              typename reduce_t = typename internal::return_type<optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    matvec(offset_t C, optr_t o, hptr_t h, iptr_t i,
           reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        auto hh = h + C;         // pointer to off-diagonal elements
        reduce_t ii = static_cast<reduce_t>(i[C-1]);    // last input element
        reduce_t oo = static_cast<reduce_t>(0);
        reduce_t hc, hhc, ic;
        for (offset_t c = 0; c < C-1; ++c, ++h, ++hh, ++i, ++o) {
            internal::set(hhc, *hh);
            internal::set(hc, *h);
            internal::set(ic, *i);
            internal::set(*o, hc * ic + hhc * ii);
            internal::iaddcmul<reduce_t>(oo, hhc, ic);
        }
        internal::iaddcmul<reduce_t>(oo, *h, ii);
        internal::set(*o, oo);
    }

    template <typename optr_t, typename iptr_t, typename hptr_t,
              typename reduce_t = typename internal::return_type<optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    addmatvec_(offset_t C, optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        auto hh = h + C;         // pointer to off-diagonal elements
        reduce_t ii = static_cast<reduce_t>(i[C-1]);    // last input element
        reduce_t oo = static_cast<reduce_t>(o[C-1]);    // last output element
        reduce_t hc, hhc, ic;
        for (offset_t c = 0; c < C-1; ++c, ++h, ++hh, ++i, ++o) {
            internal::set(hhc, *hh);
            internal::set(hc, *h);
            internal::set(ic, *i);
            internal::iadd<reduce_t>(*o, hc * ic + hhc * ii);
            internal::iaddcmul<reduce_t>(oo, hhc, ic);
        }
        internal::iaddcmul<reduce_t>(oo, *h, ii);
        internal::set(*o, oo);
    }

    template <typename optr_t, typename iptr_t, typename hptr_t,
              typename reduce_t = typename internal::return_type<optr_t, hptr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(offset_t C, optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        auto hh = h + C;         // pointer to off-diagonal elements
        reduce_t ii = static_cast<reduce_t>(i[C-1]);    // last input element
        reduce_t oo = static_cast<reduce_t>(o[C-1]);    // last output element
        reduce_t hc, hhc, ic;
        for (offset_t c = 0; c < C-1; ++c, ++h, ++hh, ++i, ++o) {
            internal::set(hhc, *hh);
            internal::set(hc, *h);
            internal::set(ic, *i);
            internal::isub<reduce_t>(*o, hc * ic + hhc * ii);
            internal::isubcmul<reduce_t>(oo, hhc, ic);
        }
        internal::isubcmul<reduce_t>(oo, *h, ii);
        internal::set(*o, oo);
    }

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_(offset_t C, vptr_t v, hptr_t h,
                wptr_t w = nullptr, bptr_t /*b*/ = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        auto hh = h + C;         // pointer to off-diagonal elements
        reduce_t ov = static_cast<reduce_t>(0), oh;
        internal::add<reduce_t>(oh, h[C-1], w[C-1]);
        for (offset_t c = 0; c < C-1; ++c, ++h, ++hh, ++v, ++w) {
            reduce_t hc = static_cast<reduce_t>(*h);
            reduce_t hhc = static_cast<reduce_t>(*hh);
            reduce_t tmp = hhc / hc;
            internal::iadd<reduce_t>(hc, *w);
            internal::isubcmul<reduce_t>(oh, hhc, tmp);
            internal::iaddcmul<reduce_t>(ov, *v, tmp);
        }
        oh = 1. / oh; // oh = 1/mini_inv, ov = sum(vec_norm * grad)
        reduce_t tmp = (static_cast<reduce_t>(*v) - ov) * oh;
        internal::set(*v, tmp);
        --v; --h; --hh;
        for (offset_t c = 0; c < C-1; ++c, --v, --h, --hh) {
            reduce_t vc = static_cast<reduce_t>(*v);
            reduce_t hc = static_cast<reduce_t>(*h);
            reduce_t hhc = static_cast<reduce_t>(*hh);
            internal::set(*v, (vc - tmp * hhc) / hc);
        }
    }
};

#endif  // JF_POSDEF_ESTATICS
