#ifndef JF_POSDEF_SYM
#define JF_POSDEF_SYM

// ---------------------------------------
// Generic implementation with static size
// ---------------------------------------
template <typename offset_t, int C>
struct utils<type::Sym, offset_t, C>: public common_sym<offset_t, C>
{
    using this_type = utils<type::Sym, offset_t, C>;
    static constexpr bool need_buffer = true;
    static constexpr offset_t work_size = C*C;

    template <typename hptr_t, typename xptr_t, typename yptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, xptr_t, yptr_t>::value>
    static inline __device__ void
    matvec_backward(hptr_t h, xptr_t x, yptr_t y,
                    reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c, ++h)
            internal::mul<reduce_t>(*h, x[c], y[c]);
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
            for (offset_t cc = c+1; cc < C; ++cc, ++h) {
                reduce_t tmp;
                internal::mul<reduce_t>(tmp, x[c], y[cc]);
                internal::iaddcmul<reduce_t>(tmp, x[cc], y[c]);
                internal::set(*h, tmp);
            }
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    matvec(optr_t o, hptr_t h, iptr_t i,
           reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c) {
            reduce_t acc = static_cast<reduce_t>(0);
#           pragma unroll
            for (offset_t cc = 0; cc < C; ++cc) {
                offset_t k = sub2pak(c, cc);
                internal::iaddcmul<reduce_t>(acc, h[k], i[cc]);
            }
            internal::set(o[c], acc);
        }
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    addmatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c) {
            reduce_t acc = static_cast<reduce_t>(0);
#           pragma unroll
            for (offset_t cc = 0; cc < C; ++cc) {
                offset_t k = sub2pak(C, c, cc);
                internal::iaddcmul<reduce_t>(acc, h[k], i[cc]);
            }
            internal::iadd<reduce_t>(o[c], acc);
        }
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c) {
            reduce_t acc = static_cast<reduce_t>(0);
#           pragma unroll
            for (offset_t cc = 0; cc < C; ++cc) {
                offset_t k = sub2pak(C, c, cc);
                internal::iaddcmul<reduce_t>(acc, h[k], i[cc]);
            }
            internal::isub<reduce_t>(o[c], acc);
        }
    }

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_(vptr_t v, hptr_t h,
                wptr_t w = nullptr, bptr_t b = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        tofull(b, h);
        if (w)
#           pragma unroll
            for (offset_t c = 0; c < C; ++c)
                internal::iadd<reduce_t>(b[c*C+c], w[c]);

        cholesky<offset_t, C>::decompose_(b, static_cast<reduce_t>(0));  // cholesky decomposition
        cholesky<offset_t, C>::solve_(b, v, static_cast<reduce_t>(0));   // solve linear system inplace
    }

    template <typename optr_t, typename hptr_t,
              typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, bptr_t>::value>
    static inline __device__
    void invert(optr_t o, hptr_t h,
                bptr_t b = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        copy_((C*(C+1))/2, o, h);
        return invert_(o, b, static_cast<reduce_t>(0));
    }

    template <typename hptr_t, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<hptr_t, bptr_t>::value>
    static inline __device__
    void invert_(hptr_t h,
                 bptr_t b = nullptr,
                 reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        // The input (and therefore output) matrices have a "diag then row"
        // layout. That is, the linear indices are:
        // [ 0  4  5  6 ]
        // [ .  1  7  8 ]
        // [ .  .  2  9 ]
        // [ .  .  .  3 ]

        // We first copy the compact matric into a full CxC buffer
        // and compute the cholesky decomposition
        tofull(b, h);
        cholesky<offset_t, C>::decompose_(b, static_cast<reduce_t>(0));  // cholesky decomposition

        // We need to solve the linear system at each basis vector [1 0 ...]
        //
        // We save results in symmetric matrix ordered by rows
        // (instead of diagonal then rows) because we can fill it
        // sequentially while using the first row to solve the system.
        // That is, the linear indices of the intermediate matrix are
        // [ 0  1  2  3 ]
        // [ .  4  5  6 ]
        // [ .  .  7  8 ]
        // [ .  .  .  9 ]
        // The first row [ 0  1  2  3 ] is used to solve the linear
        // system in-place, and the upper half of the result is then
        // copied into its correct column:
        // 1)
        // [ 0  0  0  1 ]    [ a  b  c  d ]    [ -  -  -  - ]
        // [ .  -  -  - ] -> [ .  -  -  - ] -> [ .  -  -  b ]
        // [ .  .  -  - ]    [ .  .  -  - ]    [ .  .  -  c ]
        // [ .  .  .  - ]    [ .  .  .  - ]    [ .  .  .  d ]
        // 2)
        // [ 0  0  1  0 ]    [ e  f  g  h ]    [ -  -  -  - ]
        // [ .  -  -  b ] -> [ .  -  -  b ] -> [ .  -  f  b ]
        // [ .  .  -  c ]    [ .  .  -  c ]    [ .  .  g  c ]
        // [ .  .  .  d ]    [ .  .  -  d ]    [ .  .  .  d ]
        // 3)
        // [ 0  1  0  0 ]    [ i  j  k  l ]    [ -  -  -  - ]
        // [ .  -  f  b ] -> [ .  -  f  b ] -> [ .  j  f  b ]
        // [ .  .  g  c ]    [ .  .  g  c ]    [ .  .  g  c ]
        // [ .  .  .  d ]    [ .  .  .  d ]    [ .  .  .  d ]
        // 4 == final)
        // [ 1  0  0  0 ]    [ m  n  o  p ]
        // [ .  j  f  b ] -> [ .  j  f  b ]
        // [ .  .  g  c ]    [ .  .  g  c ]
        // [ .  .  .  d ]    [ .  .  .  d ]

#       pragma unroll
        for (offset_t k = C-1; k > 0; --k)
        {
            // set basis vector in first row
#           pragma unroll
            for (offset_t c = 0; c < C; ++c)
                internal::set(h[c], c == k ? 1. : 0.);

            // solve linear system inplace
            cholesky<offset_t, C>::solve_(b, h, static_cast<reduce_t>(0));

            // copy triangle
#           pragma unroll
            for (offset_t kk = 1; kk <= k; ++kk)
                internal::set(h[sub2pak_rows(kk, k)], h[kk]);
        }

        // last row (no need to copy, we are already in the right place)
        internal::set(h[0], 1.);
#       pragma unroll
        for (offset_t c = 1; c < C; ++c)
            internal::set(h[c], 0.);
        cholesky<offset_t, C>::solve_(b, h, static_cast<reduce_t>(0));

        // now we need to shuffle the compact matrix to recover a
        // "diagonal then rows" layout. let's use the buffer
        //
        // E.g. we want to go
        // [ 0  1  2  3 ]    [ 0  4  5  6 ]
        // [ .  4  5  6 ] -> [ .  1  7  8 ]
        // [ .  .  7  8 ]    [ .  .  2  9 ]
        // [ .  .  .  9 ]    [ .  .  .  3 ]
        //
        // I don't think there's an elegant way to do that with just
        // swaps, so I'll copy the result in the buffer and then back

        // rows -> full (do not fill lower triangle)
        auto i = h;
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
#           pragma unroll
            for (offset_t cc = c; cc < C; ++cc, ++i)
                internal::set(b[c+C*cc], *i);
        // full -> diag then rows
        i = h;
#       pragma unroll
        for (offset_t c = 0; c < C; ++c, ++i)
            internal::set(*i, b[c+C*c]);
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
#           pragma unroll
            for (offset_t cc = c+1; cc < C; ++cc, ++i)
                internal::set(*i, b[c+C*cc]);
    }

    template <typename optr_t, typename iptr_t>
    static inline __device__ void
    tofull(optr_t o, iptr_t i)
    {
        using scalar_t = typename internal::elem_type<optr_t>::value;
        scalar_t foo;
#       pragma unroll
        for (offset_t c = 0; c < C; ++c, ++i)
        {
            o[C*c+c] = static_cast<scalar_t>((*i) * JFH_OnePlusTiny);
        }
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
        {
#           pragma unroll
            for (offset_t cc = c+1; cc < C; ++cc, ++i)
            {
                scalar_t ii = static_cast<scalar_t>(*i);
                o[C*cc+c] = ii;
                // o[C*c+cc] = ii; // TODO WHY IS THIS NOT DOING ANYTHING?
            }
        }

        // TODO:
        // This should not be needed but there is a bug I don't understand
        // in the lines above. Looks like an oob thing where you write over
        // the program and any line added changes the end behaviour, so
        // it seems very major and I really need to come back to it.
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
#           pragma unroll
            for (offset_t cc = c+1; cc < C; ++cc)
                o[C*c+cc] = o[C*cc+c];
    }

    template <typename optr_t, typename iptr_t>
    static inline __device__ void
    fromfull(optr_t o, iptr_t i)
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c, ++o)
            internal::set(*o, i[c*C+c]);

#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
#           pragma unroll
            for (offset_t cc = c+1; cc < C; ++cc, ++o)
                internal::set(*o, i[c*C+cc]);
    }

//protected:

    template <int K>
    static inline __device__
    offset_t sub2pak_rows(offset_t i, offset_t j)
    {
        return j < i ? sub2pak_rows<K>(j, i) : i*K - (i*(i+1)) / 2 + j;
    }

    static inline __device__
    offset_t sub2pak(offset_t i, offset_t j)
    {
        return j < i ? sub2pak(j, i) : i == j ? i : C + sub2pak_rows<C-1>(i, j-1);
    }
};


// -------------------------------
// Specialization for dynamic size
// -------------------------------
template <typename offset_t>
struct utils<type::Sym, offset_t, 0>: public common_sym<offset_t, 0>
{
    using this_type = utils<type::Sym, offset_t, 0>;
    static constexpr bool need_buffer = true;

    static inline offset_t work_size(offset_t C) { return C*C; }

    template <typename hptr_t, typename xptr_t, typename yptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, xptr_t, yptr_t>::value>
    static inline __device__ void
    matvec_backward(offset_t C, hptr_t h, xptr_t x, yptr_t y,
                    reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c, ++h)
            internal::mul<reduce_t>(*h, x[c], y[c]);
        for (offset_t c = 0; c < C; ++c)
            for (offset_t cc = c+1; cc < C; ++cc, ++h) {
                reduce_t tmp;
                internal::mul<reduce_t>(tmp, x[c], y[cc]);
                internal::iaddcmul<reduce_t>(tmp, x[cc], y[c]);
                internal::set(*h, tmp);
            }
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    matvec(offset_t C, optr_t o, hptr_t h, iptr_t i,
           reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c) {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t cc = 0; cc < C; ++cc) {
                offset_t k = sub2pak(C, c, cc);
                internal::iaddcmul<reduce_t>(acc, h[k], i[cc]);
            }
            internal::set(o[c], acc);
        }
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    addmatvec_(offset_t C, optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c) {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t cc = 0; cc < C; ++cc) {
                offset_t k = sub2pak(C, c, cc);
                internal::iaddcmul<reduce_t>(acc, h[k], i[cc]);
            }
            internal::iadd<reduce_t>(o[c], acc);
        }
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(offset_t C, optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        for (offset_t c = 0; c < C; ++c) {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t cc = 0; cc < C; ++cc) {
                offset_t k = sub2pak(C, c, cc);
                internal::iaddcmul<reduce_t>(acc, h[k], i[cc]);
            }
            internal::isub<reduce_t>(o[c], acc);
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
        tofull(C, b, h);
        if (w)
            for (offset_t c = 0; c < C; ++c)
                internal::iadd<reduce_t>(b[c*C+c], w[c]);

        cholesky<offset_t>::decompose_(C, b, static_cast<reduce_t>(0));  // cholesky decomposition
        cholesky<offset_t>::solve_(C, b, v, static_cast<reduce_t>(0));   // solve linear system inplace
    }

    template <typename optr_t, typename hptr_t,
              typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, bptr_t>::value>
    static inline __device__
    void invert(offset_t C, optr_t o, hptr_t h,
                bptr_t b = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        copy_((C*(C+1))/2, o, h);
        return invert_(C, o, b, static_cast<reduce_t>(0));
    }

    template <typename hptr_t, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<hptr_t, bptr_t>::value>
    static inline __device__
    void invert_(offset_t C, hptr_t h,
                 bptr_t b = nullptr,
                 reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        // The input (and therefore output) matrices have a "diag then row"
        // layout. That is, the linear indices are:
        // [ 0  4  5  6 ]
        // [ .  1  7  8 ]
        // [ .  .  2  9 ]
        // [ .  .  .  3 ]

        // We first copy the compact matric into a full CxC buffer
        // and compute the cholesky decomposition
        tofull(C, b, h);
        cholesky<offset_t>::decompose_(C, b, static_cast<reduce_t>(0));  // cholesky decomposition

        // We need to solve the linear system at each basis vector [1 0 ...]
        //
        // We save results in symmetric matrix ordered by rows
        // (instead of diagonal then rows) because we can fill it
        // sequentially while using the first row to solve the system.
        // That is, the linear indices of the intermediate matrix are
        // [ 0  1  2  3 ]
        // [ .  4  5  6 ]
        // [ .  .  7  8 ]
        // [ .  .  .  9 ]
        // The first row [ 0  1  2  3 ] is used to solve the linear
        // system in-place, and the upper half of the result is then
        // copied into its correct column:
        // 1)
        // [ 0  0  0  1 ]    [ a  b  c  d ]    [ -  -  -  - ]
        // [ .  -  -  - ] -> [ .  -  -  - ] -> [ .  -  -  b ]
        // [ .  .  -  - ]    [ .  .  -  - ]    [ .  .  -  c ]
        // [ .  .  .  - ]    [ .  .  .  - ]    [ .  .  .  d ]
        // 2)
        // [ 0  0  1  0 ]    [ e  f  g  h ]    [ -  -  -  - ]
        // [ .  -  -  b ] -> [ .  -  -  b ] -> [ .  -  f  b ]
        // [ .  .  -  c ]    [ .  .  -  c ]    [ .  .  g  c ]
        // [ .  .  .  d ]    [ .  .  -  d ]    [ .  .  .  d ]
        // 3)
        // [ 0  1  0  0 ]    [ i  j  k  l ]    [ -  -  -  - ]
        // [ .  -  f  b ] -> [ .  -  f  b ] -> [ .  j  f  b ]
        // [ .  .  g  c ]    [ .  .  g  c ]    [ .  .  g  c ]
        // [ .  .  .  d ]    [ .  .  .  d ]    [ .  .  .  d ]
        // 4 == final)
        // [ 1  0  0  0 ]    [ m  n  o  p ]
        // [ .  j  f  b ] -> [ .  j  f  b ]
        // [ .  .  g  c ]    [ .  .  g  c ]
        // [ .  .  .  d ]    [ .  .  .  d ]

        for (offset_t k = C-1; k > 0; --k)
        {
            // set basis vector in first row
            for (offset_t c = 0; c < C; ++c)
                internal::set(h[c], c == k ? 1. : 0.);

            // solve linear system inplace
            cholesky<offset_t>::solve_(C, b, h, static_cast<reduce_t>(0));

            // copy triangle
            for (offset_t kk = 1; kk <= k; ++kk)
                internal::set(h[sub2pak_rows(C, kk, k)], h[kk]);
        }

        // last row (no need to copy, we are already in the right place)
        internal::set(h[0], 1.);
        for (offset_t c = 1; c < C; ++c)
            internal::set(h[c], 0.);
        cholesky<offset_t>::solve_(C, b, h, static_cast<reduce_t>(0));

        // now we need to shuffle the compact matrix to recover a
        // "diagonal then rows" layout. let's use the buffer
        //
        // E.g. we want to go
        // [ 0  1  2  3 ]    [ 0  4  5  6 ]
        // [ .  4  5  6 ] -> [ .  1  7  8 ]
        // [ .  .  7  8 ]    [ .  .  2  9 ]
        // [ .  .  .  9 ]    [ .  .  .  3 ]
        //
        // I don't think there's an elegant way to do that with just
        // swaps, so I'll copy the result in the buffer and then back

        // rows -> full (do not fill lower triangle)
        auto i = h;
        for (offset_t c = 0; c < C; ++c)
            for (offset_t cc = c; cc < C; ++cc, ++i)
                internal::set(b[c+C*cc], *i);
        // full -> diag then rows
        i = h;
        for (offset_t c = 0; c < C; ++c, ++i)
                internal::set(*i, b[c+C*c]);
        for (offset_t c = 0; c < C; ++c)
            for (offset_t cc = c+1; cc < C; ++cc, ++i)
                internal::set(*i, b[c+C*cc]);
    }

    template <typename optr_t, typename iptr_t>
    static inline __device__ void
    tofull(offset_t C, optr_t o, iptr_t i)
    {
        using scalar_t = typename internal::elem_type<optr_t>::value;
        scalar_t foo;
        for (offset_t c = 0; c < C; ++c, ++i)
        {
            o[C*c+c] = static_cast<scalar_t>((*i) * JFH_OnePlusTiny);
        }
        for (offset_t c = 0; c < C; ++c)
        {
            for (offset_t cc = c+1; cc < C; ++cc, ++i)
            {
                scalar_t ii = static_cast<scalar_t>(*i);
                o[C*cc+c] = ii;
                // o[C*c+cc] = ii; // TODO WHY IS THIS NOT DOING ANYTHING?
            }
        }

        // TODO:
        // This should not be needed but there is a bug I don't understand
        // in the lines above. Looks like an oob thing where you write over
        // the program and any line added changes the end behaviour, so
        // it seems very major and I really need to come back to it.
        for (offset_t c = 0; c < C; ++c)
            for (offset_t cc = c+1; cc < C; ++cc)
                o[C*c+cc] = o[C*cc+c];
    }

    template <typename optr_t, typename iptr_t>
    static inline __device__ void
    fromfull(offset_t C, optr_t o, iptr_t i)
    {
        for (offset_t c = 0; c < C; ++c, ++o)
            internal::set(*o, i[c*C+c]);

        for (offset_t c = 0; c < C; ++c)
            for (offset_t cc = c+1; cc < C; ++cc, ++o)
                internal::set(*o, i[c*C+cc]);
    }

//protected:

    static inline __device__
    offset_t sub2pak_rows(offset_t C, offset_t i, offset_t j)
    {
        return j < i ? sub2pak_rows(C, j, i) : i*C - (i*(i+1)) / 2 + j;
    }

    static inline __device__
    offset_t sub2pak(offset_t C, offset_t i, offset_t j)
    {
        return j < i ? sub2pak(C, j, i) : i == j ? i : C + sub2pak_rows(C-1, i, j-1);
    }
};


// -----------------------------
// Specialization for static 3x3
// -----------------------------
template <typename offset_t>
struct utils<type::Sym, offset_t, 3>: public common_sym<offset_t, 3>
{
    using this_type = utils<type::Sym, offset_t, 3>;
    static constexpr bool need_buffer = false;
    static constexpr offset_t work_size = 0;

    template <typename hptr_t, typename xptr_t, typename yptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, xptr_t, yptr_t>::value>
    static inline __device__ void
    matvec_backward(hptr_t h, xptr_t x, yptr_t y,
                    reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t x0 = static_cast<reduce_t>(x[0]),
                 x1 = static_cast<reduce_t>(x[1]),
                 x2 = static_cast<reduce_t>(x[2]),
                 y0 = static_cast<reduce_t>(y[0]),
                 y1 = static_cast<reduce_t>(y[1]),
                 y2 = static_cast<reduce_t>(y[2]);

        internal::set(h[0], x0 * y0);
        internal::set(h[1], x1 * y1);
        internal::set(h[2], x2 * y2);
        internal::set(h[3], x0 * y1 + y0 * x1);
        internal::set(h[4], x0 * y2 + y0 * x2);
        internal::set(h[5], x1 * y2 + y1 * x2);
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    matvec(optr_t o, hptr_t h, iptr_t i,
           reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t x0  = static_cast<reduce_t>(i[0]),
                 x1  = static_cast<reduce_t>(i[1]),
                 x2  = static_cast<reduce_t>(i[2]),
                 h00 = static_cast<reduce_t>(h[0]),
                 h11 = static_cast<reduce_t>(h[1]),
                 h22 = static_cast<reduce_t>(h[2]),
                 h01 = static_cast<reduce_t>(h[3]),
                 h02 = static_cast<reduce_t>(h[4]),
                 h12 = static_cast<reduce_t>(h[5]);

        internal::set(o[0], h00 * x0 + h01 * x1 + h02 * x2);
        internal::set(o[1], h01 * x0 + h11 * x1 + h12 * x2);
        internal::set(o[2], h02 * x0 + h12 * x1 + h22 * x2);
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    addmatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t x0  = static_cast<reduce_t>(i[0]),
                 x1  = static_cast<reduce_t>(i[1]),
                 x2  = static_cast<reduce_t>(i[2]),
                 h00 = static_cast<reduce_t>(h[0]),
                 h11 = static_cast<reduce_t>(h[1]),
                 h22 = static_cast<reduce_t>(h[2]),
                 h01 = static_cast<reduce_t>(h[3]),
                 h02 = static_cast<reduce_t>(h[4]),
                 h12 = static_cast<reduce_t>(h[5]);

        internal::iadd<reduce_t>(o[0], h00 * x0 + h01 * x1 + h02 * x2);
        internal::iadd<reduce_t>(o[1], h01 * x0 + h11 * x1 + h12 * x2);
        internal::iadd<reduce_t>(o[2], h02 * x0 + h12 * x1 + h22 * x2);
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t x0  = static_cast<reduce_t>(i[0]),
                 x1  = static_cast<reduce_t>(i[1]),
                 x2  = static_cast<reduce_t>(i[2]),
                 h00 = static_cast<reduce_t>(h[0]),
                 h11 = static_cast<reduce_t>(h[1]),
                 h22 = static_cast<reduce_t>(h[2]),
                 h01 = static_cast<reduce_t>(h[3]),
                 h02 = static_cast<reduce_t>(h[4]),
                 h12 = static_cast<reduce_t>(h[5]);

        internal::isub<reduce_t>(o[0], h00 * x0 + h01 * x1 + h02 * x2);
        internal::isub<reduce_t>(o[1], h01 * x0 + h11 * x1 + h12 * x2);
        internal::isub<reduce_t>(o[2], h02 * x0 + h12 * x1 + h22 * x2);
    }

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_(vptr_t v, hptr_t h,
                wptr_t w = nullptr, bptr_t b = nullptr,
                reduce_t unused = static_cast<reduce_t>(0))
    {
        reduce_t x0  = static_cast<reduce_t>(v[0]),
                 x1  = static_cast<reduce_t>(v[1]),
                 x2  = static_cast<reduce_t>(v[2]),
                 h00 = static_cast<reduce_t>(h[0]) * JFH_OnePlusTiny,
                 h11 = static_cast<reduce_t>(h[1]) * JFH_OnePlusTiny,
                 h22 = static_cast<reduce_t>(h[2]) * JFH_OnePlusTiny,
                 h01 = static_cast<reduce_t>(h[3]),
                 h02 = static_cast<reduce_t>(h[4]),
                 h12 = static_cast<reduce_t>(h[5]);

        if (w)
        {
            h00 += static_cast<reduce_t>(w[0]);
            h11 += static_cast<reduce_t>(w[1]);
            h22 += static_cast<reduce_t>(w[2]);
        }

        reduce_t idt = h00*h11*h22 + 2*h01*h02*h12
                     - h00*h12*h12 - h11*h02*h02 - h22*h01*h01;
        idt = static_cast<reduce_t>(1) / idt;
        internal::set(v[0], idt*(x0*(h11*h22-h12*h12) + x1*(h02*h12-h01*h22) + x2*(h01*h12-h02*h11)));
        internal::set(v[1], idt*(x0*(h02*h12-h01*h22) + x1*(h00*h22-h02*h02) + x2*(h01*h02-h00*h12)));
        internal::set(v[2], idt*(x0*(h01*h12-h02*h11) + x1*(h01*h02-h00*h12) + x2*(h00*h11-h01*h01)));
    }

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_le_(vptr_t v, hptr_t h,
                   wptr_t w = nullptr, bptr_t b = nullptr,
                   reduce_t unused = static_cast<reduce_t>(0))
    {
        reduce_t x0  = static_cast<reduce_t>(v[0]),
                 x1  = static_cast<reduce_t>(v[1]),
                 x2  = static_cast<reduce_t>(v[2]),
                 h00 = static_cast<reduce_t>(h[0]) * JFH_OnePlusTiny,
                 h11 = static_cast<reduce_t>(h[1]) * JFH_OnePlusTiny,
                 h22 = static_cast<reduce_t>(h[2]) * JFH_OnePlusTiny,
                 h01 = static_cast<reduce_t>(h[3]),
                 h02 = static_cast<reduce_t>(h[4]),
                 h12 = static_cast<reduce_t>(h[5]);

        if (w)
        {
            h00 += static_cast<reduce_t>(w[0]);
            h11 += static_cast<reduce_t>(w[1]);
            h22 += static_cast<reduce_t>(w[2]);
        }

        reduce_t idt = h00*h11*h22 + 2*h01*h02*h12
                     - h00*h12*h12 - h11*h02*h02 - h22*h01*h01;
        idt = static_cast<reduce_t>(1) / idt;
        internal::set(v[0], idt*(x0*(h11*h22-h12*h12) + x1*(h02*h12-h01*h22) + x2*(h01*h12-h02*h11)));
        internal::set(v[1], idt*(x0*(h02*h12-h01*h22) + x1*(h00*h22-h02*h02) + x2*(h01*h02-h00*h12)));
        internal::set(v[2], idt*(x0*(h01*h12-h02*h11) + x1*(h01*h02-h00*h12) + x2*(h00*h11-h01*h01)));
    }

    template <typename optr_t, typename hptr_t,
              typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, bptr_t>::value>
    static inline __device__
    void invert(optr_t o, hptr_t h,
                bptr_t b = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        o[0] = h[0]; o[1] = h[1]; o[2] = h[2];
        o[3] = h[3]; o[4] = h[4]; o[5] = h[5];
        return invert_(o, b, static_cast<reduce_t>(0));
    }

    template <typename hptr_t, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<hptr_t, bptr_t>::value>
    static inline __device__
    void invert_(hptr_t h,
                 bptr_t /*b*/ = nullptr,
                 reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {

        reduce_t h00 = static_cast<reduce_t>(h[0]) * JFH_OnePlusTiny,
                 h11 = static_cast<reduce_t>(h[1]) * JFH_OnePlusTiny,
                 h22 = static_cast<reduce_t>(h[2]) * JFH_OnePlusTiny,
                 h01 = static_cast<reduce_t>(h[3]),
                 h02 = static_cast<reduce_t>(h[4]),
                 h12 = static_cast<reduce_t>(h[5]);

        reduce_t idt = h00*h11*h22 + 2*h01*h02*h12
                     - h00*h12*h12 - h11*h02*h02 - h22*h01*h01;
        idt = static_cast<reduce_t>(1) / idt;
        internal::set(h[0], idt*(h11*h22-h12*h12));
        internal::set(h[1], idt*(h00*h22-h02*h02));
        internal::set(h[2], idt*(h00*h11-h01*h01));
        internal::set(h[3], idt*(h02*h12-h01*h22));
        internal::set(h[4], idt*(h01*h12-h02*h11));
        internal::set(h[5], idt*(h01*h02-h12*h00));
    }

    template <typename optr_t, typename iptr_t>
    static inline __device__ void
    fromfull(optr_t o, iptr_t i)
    {
       internal::set(o[0], i[0]);
       internal::set(o[1], i[4]);
       internal::set(o[2], i[8]);
       internal::set(o[3], i[1]);
       internal::set(o[4], i[2]);
       internal::set(o[5], i[6]);
    }

//protected:

    static inline __device__
    offset_t sub2pak_rows(offset_t i, offset_t j)
    {
        if (j < i) return sub2pak_rows(j, i);
        switch (i) {
            case 0: return j;
            case 1: return j+3;
            case 2: return j+5;
        }
    }

    static inline __device__
    offset_t sub2pak(offset_t i, offset_t j)
    {
        if (j < i) return sub2pak(j, i);
        switch (i) {
            case 0: switch (j) {
                case 1:  return 3;
                case 2:  return 4;
                default: return 0;
            }
            case 1: switch (j) {
                case 2:  return 5;
                default: return 1;
            }
            default: return 2;
        }
    }
};


// -----------------------------
// Specialization for static 2x2
// -----------------------------
template <typename offset_t>
struct utils<type::Sym, offset_t, 2>: public common_sym<offset_t, 2>
{
    using this_type = utils<type::Sym, offset_t, 3>;
    static constexpr bool need_buffer = false;
    static constexpr offset_t work_size = 0;

    template <typename hptr_t, typename xptr_t, typename yptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, xptr_t, yptr_t>::value>
    static inline __device__ void
    matvec_backward(hptr_t h, xptr_t x, yptr_t y,
                    reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t x0 = static_cast<reduce_t>(x[0]),
                 x1 = static_cast<reduce_t>(x[1]),
                 y0 = static_cast<reduce_t>(y[0]),
                 y1 = static_cast<reduce_t>(y[1]);

        internal::set(h[0], x0 * y0);
        internal::set(h[1], x1 * y1);
        internal::set(h[2], x0 * y1 + y0 * x1);
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    matvec(optr_t o, hptr_t h, iptr_t i,
           reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t x0  = static_cast<reduce_t>(i[0]),
                 x1  = static_cast<reduce_t>(i[1]),
                 h00 = static_cast<reduce_t>(h[0]),
                 h11 = static_cast<reduce_t>(h[1]),
                 h01 = static_cast<reduce_t>(h[2]);

        internal::set(o[0], h00 * x0 + h01 * x1);
        internal::set(o[1], h01 * x0 + h11 * x1);
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    addmatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t x0  = static_cast<reduce_t>(i[0]),
                 x1  = static_cast<reduce_t>(i[1]),
                 h00 = static_cast<reduce_t>(h[0]),
                 h11 = static_cast<reduce_t>(h[1]),
                 h01 = static_cast<reduce_t>(h[2]);

        internal::iadd<reduce_t>(o[0], h00 * x0 + h01 * x1);
        internal::iadd<reduce_t>(o[1], h01 * x0 + h11 * x1);
    }

    template <typename optr_t, typename hptr_t, typename iptr_t,
              typename reduce_t = typename internal::return_type<hptr_t, optr_t, iptr_t>::value>
    static inline __device__ void
    submatvec_(optr_t o, hptr_t h, iptr_t i,
               reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t x0  = static_cast<reduce_t>(i[0]),
                 x1  = static_cast<reduce_t>(i[1]),
                 h00 = static_cast<reduce_t>(h[0]),
                 h11 = static_cast<reduce_t>(h[1]),
                 h01 = static_cast<reduce_t>(h[2]);

        internal::isub<reduce_t>(o[0], h00 * x0 + h01 * x1);
        internal::isub<reduce_t>(o[1], h01 * x0 + h11 * x1);
    }

    template <typename vptr_t, typename hptr_t,
              typename wptr_t = const void *, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<vptr_t, hptr_t, wptr_t, bptr_t>::value>
    static inline __device__ void
    solve_impl_(vptr_t v, hptr_t h,
                wptr_t w = nullptr, bptr_t /*b*/ = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        reduce_t x0  = static_cast<reduce_t>(v[0]),
                 x1  = static_cast<reduce_t>(v[1]),
                 h00 = static_cast<reduce_t>(h[0]) * JFH_OnePlusTiny,
                 h11 = static_cast<reduce_t>(h[1]) * JFH_OnePlusTiny,
                 h01 = static_cast<reduce_t>(h[2]);

        if (w)
        {
            h00 += static_cast<reduce_t>(w[0]);
            h11 += static_cast<reduce_t>(w[1]);
        }

        reduce_t idt = static_cast<reduce_t>(1) / (h00*h11 - h01*h01);
        internal::set(v[0], idt*(x0*h11 - x1*h01));
        internal::set(v[1], idt*(x1*h00 - x0*h01));
    }

    template <typename optr_t, typename hptr_t,
              typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<
                                  optr_t, hptr_t, bptr_t>::value>
    static inline __device__
    void invert(optr_t o, hptr_t h,
                bptr_t /*b*/ = nullptr,
                reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        o[0] = h[0]; o[1] = h[1]; o[2] = h[2];
        return invert_(o, nullptr, static_cast<reduce_t>(0));
    }

    template <typename hptr_t, typename bptr_t = const void *,
              typename reduce_t = typename internal::return_type<hptr_t, bptr_t>::value>
    static inline __device__
    void invert_(hptr_t h,
                 bptr_t /*b*/ = nullptr,
                 reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {

        reduce_t h00 = static_cast<reduce_t>(h[0]) * JFH_OnePlusTiny,
                 h11 = static_cast<reduce_t>(h[1]) * JFH_OnePlusTiny,
                 h01 = static_cast<reduce_t>(h[2]);

        reduce_t idt = static_cast<reduce_t>(1) / (h00*h11 - h01*h01);
        internal::set(h[0],  idt*h11);
        internal::set(h[1],  idt*h00);
        internal::set(h[2], -idt*h01);
    }

    template <typename optr_t, typename iptr_t>
    static inline __device__ void
    fromfull(optr_t o, iptr_t i)
    {
       internal::set(o[0], i[0]);
       internal::set(o[1], i[2]);
       internal::set(o[2], i[1]);
    }
};

#endif // JF_POSDEF_SYM
