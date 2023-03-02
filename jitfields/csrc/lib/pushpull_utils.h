#ifndef JF_PUSHPULL_UTILS
#define JF_PUSHPULL_UTILS
#include "cuda_switch.h"
#include "spline.h"
#include "bounds.h"

namespace jf {
namespace pushpull {

const spline::type Z = spline::type::Nearest;
const spline::type L = spline::type::Linear;
const spline::type Q = spline::type::Quadratic;
const spline::type C = spline::type::Cubic;
const bound::type B0 = bound::type::NoCheck;
const int zero  = 0;
const int one   = 1;
const int two   = 2;
const int three = 3;
const int mone  = -1;

template <int D,
          spline::type IX=Z,  bound::type BX=B0,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
struct PushPull {};


/***********************************************************************
 *
 *                                UTILS
 *
 **********************************************************************/


/*** Check In/Out of Bounds *******************************************/
#define JF_EXTRAPOLATE_TINY 5E-2

template <int extrapolate, int D>
struct InFOV {};

template <int D>
struct InFOV<one, D> {
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(const scalar_t * loc, const offset_t * size, offset_t stride=1) {
        return true;
    }
};

template <int D>
struct InFOV<zero, D> { // Limits at voxel centers
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(const scalar_t * loc, const offset_t * size, offset_t stride=1) {
#       pragma unroll
        for (int d=0; d < D; ++d, loc += stride) {
            scalar_t loc1 = *loc;
            if (loc1 < -JF_EXTRAPOLATE_TINY)
                return false;
            if (loc1 > size[d] - 1 + JF_EXTRAPOLATE_TINY)
                return false;
        }
        return true;
    }
};

template <int D>
struct InFOV<mone, D> { // Limits at voxel edges
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(const scalar_t * loc, const offset_t * size, offset_t stride=1) {
#       pragma unroll
        for (int d=0; d < D; ++d, loc += stride) {
            scalar_t loc1 = *loc;
            if (loc1 < - 0.5 - JF_EXTRAPOLATE_TINY)
                return false;
            if (loc1 > size[d] - 0.5 + JF_EXTRAPOLATE_TINY)
                return false;
        }
        return true;
    }
};

/*
template <>
struct InFOV<one, one> {
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(scalar_t x, offset_t nx) {
        return true;
    }
};

template <>
struct InFOV<zero, one> { // Limits at voxel centers
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(scalar_t x, offset_t nx) {
        if (x < -JF_EXTRAPOLATE_TINY)
            return false;
        if (x > nx - 1 + JF_EXTRAPOLATE_TINY)
            return false;
        return true;
    }
};

template <>
struct InFOV<mone, one> { // Limits at voxel edges
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(scalar_t x, offset_t nx) {
        if (x < -0.5 - JF_EXTRAPOLATE_TINY)
            return false;
        if (x > nx - 0.5 + JF_EXTRAPOLATE_TINY)
            return false;
        return true;
    }
};

template <>
struct InFOV<one, two> {
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(scalar_t x, scalar_t y, offset_t nx, offset_t ny) {
        return true;
    }
};

template <>
struct InFOV<zero, two> {
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(scalar_t x, scalar_t y, offset_t nx, offset_t ny) {
        return InFOV<0, 1>::infov(x, nx) &&
               InFOV<0, 1>::infov(y, ny);
    }
};

template <>
struct InFOV<mone, two> {
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(scalar_t x, scalar_t y, offset_t nx, offset_t ny) {
        return InFOV<-1, 1>::infov(x, nx) &&
               InFOV<-1, 1>::infov(y, ny);
    }
};

template <>
struct InFOV<one, three> {
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(scalar_t x, scalar_t y, scalar_t z,
          offset_t nx, offset_t ny, offset_t nz) {
        return true;
    }
};

template <>
struct InFOV<zero, three> {
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(scalar_t x, scalar_t y, scalar_t z,
          offset_t nx, offset_t ny, offset_t nz) {
        return InFOV<0, 1>::infov(x, nx) &&
               InFOV<0, 1>::infov(y, ny) &&
               InFOV<0, 1>::infov(z, nz);
    }
};

template <>
struct InFOV<mone, three> {
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(scalar_t x, scalar_t y, scalar_t z,
          offset_t nx, offset_t ny, offset_t nz) {
        return InFOV<-1, 1>::infov(x, nx) &&
               InFOV<-1, 1>::infov(y, ny) &&
               InFOV<-1, 1>::infov(z, nz);
    }
};
*/


/*** Wrap out-of-bounds indices ***************************************/
template <spline::type I=Z,  bound::type B=B0>
struct PushPullAnyUtils {
    using spline_utils = spline::utils<I>;
    using bound_utils = bound::utils<B>;

    template <typename reduce_t, typename offset_t>
    static __device__ offset_t
    index(reduce_t x, offset_t size, offset_t i[], reduce_t w[], signed char s[])
    {
        offset_t b0, b1;
        spline_utils::bounds(x, b0, b1);
        offset_t db = b1-b0;
        reduce_t    *ow = w;
        offset_t    *oi = i;
        signed char *os = s;
        for (offset_t b = b0; b <= b1; ++b) {
            reduce_t d = fabs(x - b);
            *(ow++)  = spline_utils::fastweight(d);
            *(os++)  = bound_utils::sign(b, size);
            *(oi++)  = bound_utils::index(b, size);
        }
        return db;
    }

    template <typename reduce_t, typename offset_t>
    static __device__ offset_t
    gindex(reduce_t x, offset_t size, offset_t i[], reduce_t w[], reduce_t g[], signed char s[])
    {
        offset_t b0, b1;
        spline_utils::bounds(x, b0, b1);
        offset_t db = b1-b0;
        reduce_t    *ow = w;
        reduce_t    *og = g;
        offset_t    *oi = i;
        signed char *os = s;
        for (offset_t b = b0; b <= b1; ++b) {
            reduce_t d = x - b;
            bool neg = d < 0;
            if (neg) d = -d;
            *(ow++)  = spline_utils::fastweight(d);
            *(og++)  = spline_utils::fastgrad(d) * (neg ? -1 : 1);
            *(os++)  = bound_utils::sign(b, size);
            *(oi++)  = bound_utils::index(b, size);
        }
        return db;
    }

    template <typename reduce_t, typename offset_t>
    static __device__ offset_t
    hindex(reduce_t x, offset_t size, offset_t i[],
           reduce_t w[], reduce_t g[], reduce_t h[], signed char s[])
    {
        offset_t b0, b1;
        spline_utils::bounds(x, b0, b1);
        offset_t db = b1-b0;
        reduce_t    *ow = w;
        reduce_t    *og = g;
        reduce_t    *oh = h;
        offset_t    *oi = i;
        signed char *os = s;
        for (offset_t b = b0; b <= b1; ++b) {
            reduce_t d = x - b;
            bool neg = d < 0;
            if (neg) d = -d;
            *(ow++)  = spline_utils::fastweight(d);
            *(og++)  = spline_utils::fastgrad(d) * (neg ? -1 : 1);
            *(oh++)  = spline_utils::fasthess(d);
            *(os++)  = bound_utils::sign(b, size);
            *(oi++)  = bound_utils::index(b, size);
        }
        return db;
    }
};

template <spline::type I=Z,  bound::type B=B0>
struct PushPullUtils {};

template <bound::type B>
struct PushPullUtils<Z,B> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<Z>;

    template <typename reduce_t, typename offset_t>
    static __device__ void
    index(reduce_t x, offset_t size, offset_t & ix, signed char & s)
    {
        ix = static_cast<offset_t>(round(x));
        s  = bound_utils::sign(ix, size);
        ix = bound_utils::index(ix, size);
    }
};

template <bound::type B>
struct PushPullUtils<L,B> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<L>;

    template <typename reduce_t, typename offset_t>
    static __device__ void
    index(reduce_t x, offset_t size,
          offset_t & ix0, offset_t & ix1,
          reduce_t & w0, reduce_t & w1,
          signed char & f0, signed char & f1)
    {
        ix0 = static_cast<offset_t>(floor(x));
        w1 = x - ix0;
        w0 = 1. - w1;
        f1 = bound_utils::sign(ix0+1, size);
        f0 = bound_utils::sign(ix0,   size);
        ix1 = bound_utils::index(ix0+1, size);
        ix0 = bound_utils::index(ix0,   size);
    }
};

template <bound::type B>
struct PushPullUtils<Q,B> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<Q>;

    template <typename reduce_t, typename offset_t>
    static __device__ void
    index(reduce_t x, offset_t size,
          offset_t & ix0, offset_t & ix1, offset_t & ix2,
          reduce_t & w0, reduce_t & w1, reduce_t & w2,
          signed char & f0, signed char & f1, signed char & f2)
    {
        ix1 = static_cast<offset_t>(round(x));
        ix0 = ix1 - 1;
        ix2 = ix1 + 1;
        w0 = spline_utils::fastweight(x - ix0);
        w1 = spline_utils::weight(x - ix1); // cannot use fast (sign unknown)
        w2 = spline_utils::fastweight(ix2 - x);
        f0 = bound_utils::sign(ix0, size);
        f1 = bound_utils::sign(ix1, size);
        f2 = bound_utils::sign(ix2, size);
        ix0 = bound_utils::index(ix0, size);
        ix1 = bound_utils::index(ix1, size);
        ix2 = bound_utils::index(ix2, size);
    }

    template <typename reduce_t, typename offset_t>
    static __device__ void
    gindex(reduce_t x, offset_t size,
          offset_t & ix0, offset_t & ix1, offset_t & ix2,
          reduce_t & w0, reduce_t & w1, reduce_t & w2,
          reduce_t & g0, reduce_t & g1, reduce_t & g2,
          signed char & f0, signed char & f1, signed char & f2)
    {
        ix1 = static_cast<offset_t>(round(x));
        ix0 = ix1 - 1;
        ix2 = ix1 + 1;
        w0 = spline_utils::fastweight(x - ix0);
        w1 = spline_utils::weight(x - ix1); // cannot use fast (sign unknown)
        w2 = spline_utils::fastweight(ix2 - x);
        g0 = spline_utils::fastgrad(x - ix0);
        g1 = spline_utils::grad(x - ix1); // cannot use fast (sign unknown)
        g2 = -spline_utils::fastgrad(ix2 - x);
        f0 = bound_utils::sign(ix0, size);
        f1 = bound_utils::sign(ix1, size);
        f2 = bound_utils::sign(ix2, size);
        ix0 = bound_utils::index(ix0, size);
        ix1 = bound_utils::index(ix1, size);
        ix2 = bound_utils::index(ix2, size);
    }

    template <typename reduce_t, typename offset_t>
    static __device__ void
    hindex(reduce_t x, offset_t size,
          offset_t & ix0, offset_t & ix1, offset_t & ix2,
          reduce_t & w0, reduce_t & w1, reduce_t & w2,
          reduce_t & g0, reduce_t & g1, reduce_t & g2,
          reduce_t & h0, reduce_t & h1, reduce_t & h2,
          signed char & f0, signed char & f1, signed char & f2)
    {
        ix1 = static_cast<offset_t>(round(x));
        ix0 = ix1 - 1;
        ix2 = ix1 + 1;
        w0 = spline_utils::fastweight(x - ix0);
        w1 = spline_utils::weight(x - ix1); // cannot use fast (sign unknown)
        w2 = spline_utils::fastweight(ix2 - x);
        g0 = spline_utils::fastgrad(x - ix0);
        g1 = spline_utils::grad(x - ix1); // cannot use fast (sign unknown)
        g2 = -spline_utils::fastgrad(ix2 - x);
        h0 = spline_utils::fasthess(x - ix0);
        h1 = spline_utils::hess(x - ix1); // cannot use fast (sign unknown)
        h2 = spline_utils::fasthess(ix2 - x);
        f0 = bound_utils::sign(ix0, size);
        f1 = bound_utils::sign(ix1, size);
        f2 = bound_utils::sign(ix2, size);
        ix0 = bound_utils::index(ix0, size);
        ix1 = bound_utils::index(ix1, size);
        ix2 = bound_utils::index(ix2, size);
    }
};

template <bound::type B>
struct PushPullUtils<C,B> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<C>;

    template <typename reduce_t, typename offset_t>
    static __device__ void
    index(reduce_t x, offset_t size,
          offset_t & ix0, offset_t & ix1, offset_t & ix2, offset_t & ix3,
          reduce_t & w0, reduce_t & w1, reduce_t & w2, reduce_t & w3,
          signed char & f0, signed char & f1, signed char & f2, signed char & f3)
    {
        ix1 = static_cast<offset_t>(floor(x));
        ix0 = ix1 - 1;
        ix2 = ix1 + 1;
        ix3 = ix1 + 2;
        w0 = spline_utils::fastweight(x - ix0);
        w1 = spline_utils::fastweight(x - ix1);
        w2 = spline_utils::fastweight(ix2 - x);
        w3 = spline_utils::fastweight(ix3 - x);
        f0 = bound_utils::sign(ix0, size);
        f1 = bound_utils::sign(ix1, size);
        f2 = bound_utils::sign(ix2, size);
        f3 = bound_utils::sign(ix3, size);
        ix0 = bound_utils::index(ix0, size);
        ix1 = bound_utils::index(ix1, size);
        ix2 = bound_utils::index(ix2, size);
        ix3 = bound_utils::index(ix3, size);
    }

    template <typename reduce_t, typename offset_t>
    static __device__ void
    gindex(reduce_t x, offset_t size,
          offset_t & ix0, offset_t & ix1, offset_t & ix2, offset_t & ix3,
          reduce_t & w0, reduce_t & w1, reduce_t & w2, reduce_t & w3,
          reduce_t & g0, reduce_t & g1, reduce_t & g2, reduce_t & g3,
          signed char & f0, signed char & f1, signed char & f2, signed char & f3)
    {
        ix1 = static_cast<offset_t>(floor(x));
        ix0 = ix1 - 1;
        ix2 = ix1 + 1;
        ix3 = ix1 + 2;
        w0 = spline_utils::fastweight(x - ix0);
        w1 = spline_utils::fastweight(x - ix1);
        w2 = spline_utils::fastweight(ix2 - x);
        w3 = spline_utils::fastweight(ix3 - x);
        g0 = spline_utils::fastgrad(x - ix0);
        g1 = spline_utils::fastgrad(x - ix1);
        g2 = -spline_utils::fastgrad(ix2 - x);
        g3 = -spline_utils::fastgrad(ix3 - x);
        f0 = bound_utils::sign(ix0, size);
        f1 = bound_utils::sign(ix1, size);
        f2 = bound_utils::sign(ix2, size);
        f3 = bound_utils::sign(ix3, size);
        ix0 = bound_utils::index(ix0, size);
        ix1 = bound_utils::index(ix1, size);
        ix2 = bound_utils::index(ix2, size);
        ix3 = bound_utils::index(ix3, size);
    }

    template <typename reduce_t, typename offset_t>
    static __device__ void
    hindex(reduce_t x, offset_t size,
          offset_t & ix0, offset_t & ix1, offset_t & ix2, offset_t & ix3,
          reduce_t & w0, reduce_t & w1, reduce_t & w2, reduce_t & w3,
          reduce_t & g0, reduce_t & g1, reduce_t & g2, reduce_t & g3,
          reduce_t & h0, reduce_t & h1, reduce_t & h2, reduce_t & h3,
          signed char & f0, signed char & f1, signed char & f2, signed char & f3)
    {
        ix1 = static_cast<offset_t>(floor(x));
        ix0 = ix1 - 1;
        ix2 = ix1 + 1;
        ix3 = ix1 + 2;
        w0 = spline_utils::fastweight(x - ix0);
        w1 = spline_utils::fastweight(x - ix1);
        w2 = spline_utils::fastweight(ix2 - x);
        w3 = spline_utils::fastweight(ix3 - x);
        g0 = spline_utils::fastgrad(x - ix0);
        g1 = spline_utils::fastgrad(x - ix1);
        g2 = -spline_utils::fastgrad(ix2 - x);
        g3 = -spline_utils::fastgrad(ix3 - x);
        h0 = spline_utils::fasthess(x - ix0);
        h1 = spline_utils::fasthess(x - ix1);
        h2 = spline_utils::fasthess(ix2 - x);
        h3 = spline_utils::fasthess(ix3 - x);
        f0 = bound_utils::sign(ix0, size);
        f1 = bound_utils::sign(ix1, size);
        f2 = bound_utils::sign(ix2, size);
        f3 = bound_utils::sign(ix3, size);
        ix0 = bound_utils::index(ix0, size);
        ix1 = bound_utils::index(ix1, size);
        ix2 = bound_utils::index(ix2, size);
        ix3 = bound_utils::index(ix3, size);
    }
};

} // namespace pushpull
} // namespace jf

#endif // JF_PUSHPULL_UTILS
