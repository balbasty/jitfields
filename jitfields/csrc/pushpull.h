#ifndef JF_PUSHPULL
#define JF_PUSHPULL
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
    infov(const scalar_t * loc, const offset_t * size, offset_t stride) {
        return true;
    }
};

template <int D>
struct InFOV<zero, D> { // Limits at voxel centers
    template <typename scalar_t, typename offset_t>
    static __device__ bool
    infov(const scalar_t * loc, const offset_t * size, offset_t stride) {
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
    infov(const scalar_t * loc, const offset_t * size, offset_t stride) {
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

/***********************************************************************
 *
 *                                  1D
 *
 **********************************************************************/

/***                            NEAREST                             ***/
template <bound::type B> struct PushPull<one, Z, B> {
    using bound_utils = bound::utils<B>;
    using utils = PushPullUtils<Z, B>;
    using self = PushPull<one, Z, B>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t ix;
        signed char fx;
        utils::index(x, nx, ix, fx);
        ix *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = bound::get(inp, ix, fx);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t ix;
        signed char fx;
        utils::index(x, nx, ix, fx);
        ix *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            bound::add(out, ix, *inp, fx);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out, reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t ix;
        signed char fx;
        utils::index(x, nx, ix, fx);
        ix *= sx;

        bound::add(out, ix, 1, fx);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        *gout = static_cast<scalar_t>(0);
        self::push(out, ginp, x, nx, isx, nc, osc, isc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        *gout = static_cast<scalar_t>(0);
        self::pull(out, ginp, x, nx, sx, nc, osc, isc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count_backward(scalar_t * gout, scalar_t * inp,
                        reduce_t x, offset_t nx, offset_t sx)
    {
        *gout = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        *gout = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }
};

/***                            LINEAR                              ***/
template <bound::type B> struct PushPull<one, L, B> {
    using bound_utils = bound::utils<B>;
    using utils = PushPullUtils<L, B>;
    using self = PushPull<one, L, B>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t x0, x1;
        reduce_t w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      static_cast<reduce_t>(bound::get(inp, x0, f0)) * w0
                    + static_cast<reduce_t>(bound::get(inp, x1, f1)) * w1);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, x0, val * w0, f0);
            bound::add(out, x1, val * w1, f1);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out, reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        bound::add(out, x0, w0, f0);
        bound::add(out, x1, w1, f1);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x1, f1)
                    - bound::cget<reduce_t>(inp, x0, f0));
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * w0, f0);
            bound::add(out, x1 * osx, gval * w1, f1);
            // compute input spatial gradient
            acc += gval * (bound::cget<reduce_t>(inp, x1 * isx, f1)
                         - bound::cget<reduce_t>(inp, x0 * isx, f0));
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        reduce_t acc = 0;
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // pull incoming gradient
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(ginp, x0, f0) * w0
                    + bound::cget<reduce_t>(ginp, x1, f1) * w1);
            // compute input spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            acc += val * (bound::cget<reduce_t>(ginp, x1, f1)
                        - bound::cget<reduce_t>(ginp, x0, f0));
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count_backward(scalar_t * gout, scalar_t * ginp,
                        reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t    x0, x1;
        reduce_t    w0, w1;
        signed char f0, f1;
        utils::index(x, nx, x0, x1, w0, w1, f0, f1);
        x0 *= sx;
        x1 *= sx;

        // compute input spatial gradient
        *gout = static_cast<scalar_t>(bound::cget<reduce_t>(ginp, x1, f1)
                                    - bound::cget<reduce_t>(ginp, x0, f0));
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        *gout = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }
};

/***                          QUADRATIC                             ***/
template <bound::type B> struct PushPull<one, Q, B> {
    using bound_utils = bound::utils<B>;
    using utils = PushPullUtils<Q, B>;
    using spline_utils = spline::utils<Q>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t x0, x1, x2;
        reduce_t w0, w1, w2;
        signed char   f0, f1, f2;
        utils::index(x, nx, x0, x1, x2, w0, w1, w2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * w0
                    + bound::cget<reduce_t>(inp, x1, f1) * w1
                    + bound::cget<reduce_t>(inp, x2, f2) * w2);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    x0, x1, x2;
        reduce_t    w0, w1, w2;
        signed char f0, f1, f2;
        utils::index(x, nx, x0, x1, x2, w0, w1, w2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, x0, val * w0, f0);
            bound::add(out, x1, val * w1, f1);
            bound::add(out, x2, val * w2, f2);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out, reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t x0, x1, x2;
        reduce_t w0, w1, w2;
        signed char f0, f1, f2;
        utils::index(x, nx, x0, x1, x2, w0, w1, w2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        bound::add(out, x0, w0, f0);
        bound::add(out, x1, w1, f1);
        bound::add(out, x2, w2, f2);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    x0, x1, x2;
        reduce_t    w0, w1, w2;
        reduce_t    g0, g1, g2;
        signed char f0, f1, f2;
        utils::gindex(x, nx, x0, x1, x2, w0, w1, w2, g0, g1, g2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * g0
                    + bound::cget<reduce_t>(inp, x1, f1) * g1
                    + bound::cget<reduce_t>(inp, x2, f2) * g2);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t    x0, x1, x2;
        reduce_t    w0, w1, w2;
        reduce_t    g0, g1, g2;
        signed char f0, f1, f2;
        utils::gindex(x, nx, x0, x1, x2, w0, w1, w2, g0, g1, g2, f0, f1, f2);

        reduce_t acc = 0;
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * w0, f0);
            bound::add(out, x1 * osx, gval * w1, f1);
            bound::add(out, x2 * osx, gval * w2, f2);
            // compute input spatial gradient
            acc += gval * (bound::cget<reduce_t>(inp, x0 * isx, f0) * g0
                         + bound::cget<reduce_t>(inp, x1 * isx, f1) * g1
                         + bound::cget<reduce_t>(inp, x2 * isx, f2) * g2);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t    x0, x1, x2;
        reduce_t    w0, w1, w2;
        reduce_t    g0, g1, g2;
        signed char f0, f1, f2;
        utils::gindex(x, nx, x0, x1, x2, w0, w1, w2, g0, g1, g2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        reduce_t acc = 0;
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t ginp0 = bound::cget<reduce_t>(ginp, x0, f0);
            reduce_t ginp1 = bound::cget<reduce_t>(ginp, x1, f1);
            reduce_t ginp2 = bound::cget<reduce_t>(ginp, x2, f2);
            // pull incoming gradient
            *out = static_cast<scalar_t>(ginp0 * w0 + ginp1 * w1 + ginp2 * w2);
            // compute incoming gradient spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            acc += val * (ginp0 * g0 + ginp1 * g1 + ginp2 * g2);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count_backward(scalar_t * gout, scalar_t * ginp,
                        reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t x0, x1, x2;
        reduce_t w0, w1, w2;
        reduce_t g0, g1, g2;
        signed char f0, f1, f2;
        utils::gindex(x, nx, x0, x1, x2, w0, w1, w2, g0, g1, g2, f0, f1, f2);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;

        // compute input spatial gradient
        *gout = bound::cget<reduce_t>(ginp, x0, f0) * g0
              + bound::cget<reduce_t>(ginp, x1, f1) * g1
              + bound::cget<reduce_t>(ginp, x2, f2) * g2;
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t x0, x1, x2;
        reduce_t w0, w1, w2;
        reduce_t g0, g1, g2;
        reduce_t h0, h1, h2;
        signed char f0, f1, f2;
        utils::hindex(x, nx, x0, x1, x2, w0, w1, w2, g0, g1, g2, h0, h1, h2,
                      f0, f1, f2);

        reduce_t acc = 0;
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, static_cast<scalar_t>(gval * g0), f0);
            bound::add(out, x1 * osx, static_cast<scalar_t>(gval * g1), f1);
            bound::add(out, x2 * osx, static_cast<scalar_t>(gval * g2), f2);
            // compute input spatial hessian
            acc += gval * (static_cast<reduce_t>(bound::get(inp, x0 * isx, f0)) * h0
                         + static_cast<reduce_t>(bound::get(inp, x1 * isx, f1)) * h1
                         + static_cast<reduce_t>(bound::get(inp, x2 * isx, f2)) * h2);
        }
        *gout = static_cast<scalar_t>(acc);
    }
};

/***                             CUBIC                              ***/
template <bound::type B> struct PushPull<one, C, B> {
    using bound_utils = bound::utils<B>;
    using utils = PushPullUtils<C, B>;
    using spline_utils = spline::utils<C>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        signed char   f0, f1, f2, f3;
        utils::index(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * w0
                    + bound::cget<reduce_t>(inp, x1, f1) * w1
                    + bound::cget<reduce_t>(inp, x2, f2) * w2
                    + bound::cget<reduce_t>(inp, x3, f3) * w3);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        signed char   f0, f1, f2, f3;
        utils::index(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, x0, val * w0, f0);
            bound::add(out, x1, val * w1, f1);
            bound::add(out, x2, val * w2, f2);
            bound::add(out, x3, val * w3, f3);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out, reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        signed char   f0, f1, f2, f3;
        utils::index(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        bound::add(out, x0, w0, f0);
        bound::add(out, x1, w1, f1);
        bound::add(out, x2, w2, f2);
        bound::add(out, x3, w3, f3);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        reduce_t g0, g1, g2, g3;
        signed char f0, f1, f2, f3;
        utils::gindex(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, g0, g1, g2, g3,
                      f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * g0
                    + bound::cget<reduce_t>(inp, x1, f1) * g1
                    + bound::cget<reduce_t>(inp, x2, f2) * g2
                    + bound::cget<reduce_t>(inp, x3, f3) * g3);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        reduce_t g0, g1, g2, g3;
        signed char f0, f1, f2, f3;
        utils::gindex(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, g0, g1, g2, g3,
                      f0, f1, f2, f3);

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * w0, f0);
            bound::add(out, x1 * osx, gval * w1, f1);
            bound::add(out, x2 * osx, gval * w2, f2);
            bound::add(out, x3 * osx, gval * w3, f3);
            // compute input spatial gradient
            acc += gval * (bound::cget<reduce_t>(inp, x0 * isx, f0) * g0
                         + bound::cget<reduce_t>(inp, x1 * isx, f1) * g1
                         + bound::cget<reduce_t>(inp, x2 * isx, f2) * g2
                         + bound::cget<reduce_t>(inp, x3 * isx, f3) * g3);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        reduce_t g0, g1, g2, g3;
        signed char f0, f1, f2, f3;
        utils::gindex(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, g0, g1, g2, g3,
                      f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t ginp0 = bound::cget<reduce_t>(ginp, x0, f0);
            reduce_t ginp1 = bound::cget<reduce_t>(ginp, x1, f1);
            reduce_t ginp2 = bound::cget<reduce_t>(ginp, x2, f2);
            reduce_t ginp3 = bound::cget<reduce_t>(ginp, x3, f3);
            // pull incoming gradient
            *out = static_cast<scalar_t>(ginp0 * w0 + ginp1 * w1 +
                                         ginp2 * w2 + ginp3 * w3);
            // compute incoming gradient spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            acc += val * (ginp0 * g0 + ginp1 * g1 + ginp2 * g2 + ginp3 * g3);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count_backward(scalar_t * gout, scalar_t * ginp,
                        reduce_t x, offset_t nx, offset_t sx)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        reduce_t g0, g1, g2, g3;
        signed char f0, f1, f2, f3;
        utils::gindex(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, g0, g1, g2, g3,
                      f0, f1, f2, f3);
        x0 *= sx;
        x1 *= sx;
        x2 *= sx;
        x3 *= sx;

        // compute input spatial gradient
        *gout = bound::cget<reduce_t>(ginp, x0, f0) * g0
              + bound::cget<reduce_t>(ginp, x1, f1) * g1
              + bound::cget<reduce_t>(ginp, x2, f2) * g2
              + bound::cget<reduce_t>(ginp, x3, f3) * g3;
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        offset_t x0, x1, x2, x3;
        reduce_t w0, w1, w2, w3;
        reduce_t g0, g1, g2, g3;
        reduce_t h0, h1, h2, h3;
        signed char f0, f1, f2, f3;
        utils::hindex(x, nx, x0, x1, x2, x3, w0, w1, w2, w3, g0, g1, g2, g3,
                      h0, h1, h2, h3, f0, f1, f2, f3);

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * g0, f0);
            bound::add(out, x1 * osx, gval * g1, f1);
            bound::add(out, x2 * osx, gval * g2, f2);
            bound::add(out, x3 * osx, gval * g3, f3);
            // compute input spatial hessian
            acc += gval * (bound::cget<reduce_t>(inp, x0 * isx, f0) * h0
                         + bound::cget<reduce_t>(inp, x1 * isx, f1) * h1
                         + bound::cget<reduce_t>(inp, x2 * isx, f2) * h2
                         + bound::cget<reduce_t>(inp, x3 * isx, f3) * h3);
        }
        *gout = static_cast<scalar_t>(acc);
    }
};

/***                             ANY                                ***/
template <spline::type I, bound::type B>
struct PushPull<one, I, B> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<I>;
    using utils = PushPullAnyUtils<I, B>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        offset_t length = utils::index(x, nx, ix, wx, fx);
        for (offset_t i = 0; i <= length; ++i)
            ix[i] *= sx;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= length; ++i)
                acc += bound::cget<reduce_t>(inp, ix[i], fx[i]) * wx[i];
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        offset_t length = utils::index(x, nx, ix, wx, fx);
        for (offset_t i = 0; i <= length; ++i)
            ix[i] *= sx;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            for (offset_t i = 0; i <= length; ++i)
                bound::add(out, ix[i], val * wx[i], fx[i]);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out, reduce_t x, offset_t nx, offset_t sx)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        offset_t length = utils::index(x, nx, ix, wx, fx);

        for (offset_t i = 0; i <= length; ++i)
            bound::add(out, ix[i] * sx, static_cast<scalar_t>(wx[i]), fx[i]);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        reduce_t    gx[8];
        offset_t length = utils::gindex(x, nx, ix, wx, gx, fx);
        for (offset_t i = 0; i <= length; ++i)
            ix[i] *= sx;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= length; ++i)
                acc += static_cast<reduce_t>(bound::get(inp, ix[i], fx[i])) * gx[i];
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        reduce_t    gx[8];
        offset_t length = utils::gindex(x, nx, ix, wx, gx, fx);

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t gval = static_cast<reduce_t>(*ginp);
            reduce_t acc1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= length; ++i) {
                // push incoming gradient
                bound::add(out, ix[i] * osx, gval * wx[i], fx[i]);
                // compute input spatial gradient
                acc1 += bound::cget<reduce_t>(inp, ix[i] * isx, fx[i]) * gx[i];
            }
            acc += gval * acc1;
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        reduce_t    gx[8];
        offset_t length = utils::gindex(x, nx, ix, wx, gx, fx);
        for (offset_t i = 0; i <= length; ++i)
            ix[i] *= sx;

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t val = static_cast<reduce_t>(*inp);
            reduce_t acc1 = static_cast<reduce_t>(0);
            reduce_t acc2 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= length; ++i) {
                reduce_t gval = bound::cget<reduce_t>(ginp, ix[i], fx[i]);
                // pull incoming gradient
                acc1 += gval * wx[i];
                // compute incoming gradient spatial gradient
                acc2 += gval * gx[i];
            }
            *out = static_cast<scalar_t>(acc1);
            acc += val * acc2;
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count_backward(scalar_t * gout, scalar_t * ginp,
                        reduce_t x, offset_t nx, offset_t sx)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        reduce_t    gx[8];
        offset_t length = utils::gindex(x, nx, ix, wx, gx, fx);
        for (offset_t i = 0; i <= length; ++i)
            ix[i] *= sx;

        // compute input spatial gradient
        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t i = 0; i <= length; ++i)
            acc += static_cast<reduce_t>(bound::get(ginp, ix[i], fx[i])) * gx[i];
        *gout = static_cast<scalar_t>(acc);
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        // Precompute weights and indices
        offset_t    ix[8];
        reduce_t    wx[8];
        signed char fx[8];
        reduce_t    gx[8];
        reduce_t    hx[8];
        offset_t length = utils::hindex(x, nx, ix, wx, gx, hx, fx);

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t gval = static_cast<reduce_t>(*ginp);
            reduce_t acc1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= length; ++i) {
                // push incoming gradient
                bound::add(out, ix[i] * osx, gval * gx[i], fx[i]);
                // compute input spatial hessian
                acc1 += bound::cget<reduce_t>(inp, ix[i] * isx, fx[i]) * hx[i];
            }
            acc += gval * acc1;
        }
        *gout = static_cast<scalar_t>(acc);
    }
};

/***********************************************************************
 *
 *                                  2D
 *
 **********************************************************************/

/***                            NEAREST                             ***/
template <bound::type BX, bound::type BY>
struct PushPull<two, Z, BX, Z, BY> {
    using utils_x = PushPullUtils<Z, BX>;
    using utils_y = PushPullUtils<Z, BY>;
    using self = PushPull<two, Z, BX, Z, BY>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              reduce_t y, offset_t ny, offset_t sy,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix, iy;
        signed char fx, fy;
        utils_x::index(x, nx, ix, fx);
        utils_y::index(y, ny, iy, fy);
        offset_t    i = ix * sx + iy * sy;
        signed char f = fx * fy;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = bound::get(inp, i, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              reduce_t y, offset_t ny, offset_t sy,
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix, iy;
        signed char fx, fy;
        utils_x::index(x, nx, ix, fx);
        utils_y::index(y, ny, iy, fy);
        offset_t    i = ix * sx + iy * sy;
        signed char f = fx * fy;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            bound::add(out, i, *inp, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out,
               reduce_t x, offset_t nx, offset_t sx,
               reduce_t y, offset_t ny, offset_t sy)
    {
        offset_t    ix, iy;
        signed char fx, fy;
        utils_x::index(x, nx, ix, fx);
        utils_y::index(y, ny, iy, fy);
        offset_t    i = ix * sx + iy * sy;
        signed char f = fx * fy;

        bound::add(out, i, static_cast<scalar_t>(1), f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              reduce_t y, offset_t ny, offset_t sy,
              offset_t nc, offset_t osc, offset_t isc)
    {
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout, scalar_t * inp,
                       reduce_t x, offset_t nx, offset_t sx,
                       reduce_t y, offset_t ny, offset_t sy,
                       offset_t nc, offset_t osc, offset_t isc)
    {
        *gout = static_cast<scalar_t>(0);
        self::push(out, inp, x, nx, sx, y, ny, sy, nc, osc, isc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       reduce_t y, offset_t ny, offset_t sy,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        *gout = static_cast<scalar_t>(0);
        self::pull(out, inp, x, nx, sx, y, ny, sy, nc, osc, isc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count_backward(scalar_t * gout, scalar_t * inp,
                        reduce_t x, offset_t nx, offset_t sx,
                        reduce_t y, offset_t ny, offset_t sy)
    {
        *gout = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       reduce_t y, offset_t ny, offset_t osy, offset_t isy,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc)
    {
        *gout = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }
};

#if 0
/***                            LINEAR                              ***/
template <bound::type BX, bound::type BY>
struct PushPull<two, L, BX, L, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using spline_utils = spline::utils<L>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nx, offset_t sx, reduce_t wscl,
                offset_t h, offset_t ny, offset_t sy, reduce_t hscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        offset_t ix0 = static_cast<offset_t>(floor(x));
        offset_t iy0 = static_cast<offset_t>(floor(y));
        offset_t ix1 = ix0 + 1;
        offset_t iy1 = iy0 + 1;
        reduce_t dx1 = x - ix0;
        reduce_t dy1 = y - iy0;
        reduce_t dx0 = 1 - dx1;
        reduce_t dy0 = 1 - dy1;
        signed char  fx0 = bound_utils_x::sign(ix0, nx);
        signed char  fy0 = bound_utils_y::sign(iy0, ny);
        signed char  fx1 = bound_utils_x::sign(ix1, nx);
        signed char  fy1 = bound_utils_y::sign(iy1, ny);
        ix0 = bound_utils_x::index(ix0, nx) * sx;
        iy0 = bound_utils_y::index(iy0, ny) * sy;
        ix1 = bound_utils_x::index(ix1, nx) * sx;
        iy1 = bound_utils_y::index(iy1, ny) * sy;

        auto accum1d = [ix0, ix1, dx0, dx1, fx0, fx1, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * fx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * fx1)) * dx1;
        };

        *out = static_cast<scalar_t>(accum1d(iy0, fy0) * dy0
                                   + accum1d(iy1, fy1) * dy1);
    }
};

/***                          QUADRATIC                             ***/
template <bound::type BX, bound::type BY>
struct PushPull<two, Q, BX, Q, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using spline_utils = spline::utils<Q>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nx, offset_t sx, reduce_t wscl,
                offset_t h, offset_t ny, offset_t sy, reduce_t hscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x+0.5));
        offset_t iy1 = static_cast<offset_t>(floor(y+0.5));
        reduce_t dx1 = spline_utils::weight(x - ix1);
        reduce_t dy1 = spline_utils::weight(y - iy1);
        reduce_t dx0 = spline_utils::fastweight(x - (ix1 - 1));
        reduce_t dy0 = spline_utils::fastweight(y - (iy1 - 1));
        reduce_t dx2 = spline_utils::fastweight((ix1 + 1) - x);
        reduce_t dy2 = spline_utils::fastweight((iy1 + 1) - y);
        signed char fx0 = bound_utils_x::sign(ix1-1, nx);
        signed char fy0 = bound_utils_y::sign(iy1-1, ny);
        signed char fx2 = bound_utils_x::sign(ix1+1, nx);
        signed char fy2 = bound_utils_y::sign(iy1+1, ny);
        signed char fx1 = bound_utils_x::sign(ix1,   nx);
        signed char fy1 = bound_utils_y::sign(iy1,   ny);
        offset_t ix0, iy0, ix2, iy2;
        ix0 = bound_utils_x::index(ix1-1, nx) * sx;
        iy0 = bound_utils_y::index(iy1-1, ny) * sy;
        ix2 = bound_utils_x::index(ix1+1, nx) * sx;
        iy2 = bound_utils_y::index(iy1+1, ny) * sy;
        ix1 = bound_utils_x::index(ix1,   nx) * sx;
        iy1 = bound_utils_y::index(iy1,   ny) * sy;

        auto accum1d = [ix0, ix1, ix2, dx0, dx1, dx2, fx0, fx1, fx2, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * fx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * fx1)) * dx1
               + static_cast<reduce_t>(bound::get(inp, i + ix2, s * fx2)) * dx2;
        };

        *out = static_cast<scalar_t>(accum1d(iy0, fy0) * dy0
                                   + accum1d(iy1, fy1) * dy1
                                   + accum1d(iy2, fy2) * dy2);
    }
};

/***                            CUBIC                               ***/
template <bound::type BX, bound::type BY>
struct PushPull<two, C, BX, C, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using spline_utils = spline::utils<C>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nx, offset_t sx, reduce_t wscl,
                offset_t h, offset_t ny, offset_t sy, reduce_t hscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x));
        offset_t iy1 = static_cast<offset_t>(floor(y));
        reduce_t dx1 = spline_utils::fastweight(x - ix1);
        reduce_t dy1 = spline_utils::fastweight(y - iy1);
        reduce_t dx0 = spline_utils::fastweight(x - (ix1 - 1));
        reduce_t dy0 = spline_utils::fastweight(y - (iy1 - 1));
        reduce_t dx2 = spline_utils::fastweight((ix1 + 1) - x);
        reduce_t dy2 = spline_utils::fastweight((iy1 + 1) - y);
        reduce_t dx3 = spline_utils::fastweight((ix1 + 2) - x);
        reduce_t dy3 = spline_utils::fastweight((iy1 + 2) - y);
        signed char  fx0 = bound_utils_x::sign(ix1-1, nx);
        signed char  fy0 = bound_utils_y::sign(iy1-1, ny);
        signed char  fx2 = bound_utils_x::sign(ix1+1, nx);
        signed char  fy2 = bound_utils_y::sign(iy1+1, ny);
        signed char  fx3 = bound_utils_x::sign(ix1+2, nx);
        signed char  fy3 = bound_utils_y::sign(iy1+2, ny);
        signed char  fx1 = bound_utils_x::sign(ix1,   nx);
        signed char  fy1 = bound_utils_y::sign(iy1,   ny);
        offset_t ix0, ix2, ix3, iy0, iy2, iy3;
        ix0 = bound_utils_x::index(ix1-1, nx) * sx;
        iy0 = bound_utils_y::index(iy1-1, ny) * sy;
        ix2 = bound_utils_x::index(ix1+1, nx) * sx;
        iy2 = bound_utils_y::index(iy1+1, ny) * sy;
        ix3 = bound_utils_x::index(ix1+2, nx) * sx;
        iy3 = bound_utils_y::index(iy1+2, ny) * sy;
        ix1 = bound_utils_x::index(ix1,   nx) * sx;
        iy1 = bound_utils_y::index(iy1,   ny) * sy;

        auto accum1d = [ix0, ix1, ix2, ix3, dx0, dx1, dx2, dx3,
                        fx0, fx1, fx2, fx3, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * fx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * fx1)) * dx1
               + static_cast<reduce_t>(bound::get(inp, i + ix2, s * fx2)) * dx2
               + static_cast<reduce_t>(bound::get(inp, i + ix3, s * fx3)) * dx3;
        };

        *out = static_cast<scalar_t>(accum1d(iy0, fy0) * dy0
                                   + accum1d(iy1, fy1) * dy1
                                   + accum1d(iy2, fy2) * dy2
                                   + accum1d(iy3, fy3) * dy3);
    }
};

/***                             ANY                                ***/
template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY>
struct PushPull<two, IX, BX, IY, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using spline_utils_x = spline::utils<IX>;
    using spline_utils_y = spline::utils<IY>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nx, offset_t sx, reduce_t wscl,
                offset_t h, offset_t ny, offset_t sy, reduce_t hscl,
                reduce_t shift)
    {
        // Precompute weights and indices
        reduce_t x = wscl * (w + shift) - shift;
        reduce_t y = hscl * (h + shift) - shift;
        offset_t bx0, bx1, by0, by1;
        spline_utils_x::bounds(x, bx0, bx1);
        spline_utils_y::bounds(y, by0, by1);
        offset_t dbx = bx1-bx0;
        offset_t dby = by1-by0;
        reduce_t    wx[8],  wy[8];
        offset_t    ix[8],  iy[8];
        signed char fx[8],  fy[8];
        {
            reduce_t    *owy = wy;
            offset_t    *oiy = iy;
            signed char *ofy = fy;
            for (offset_t by = by0; by <= by1; ++by) {
                scalar_t dy = y - by;
                *(owy++)  = spline_utils_y::fastweight(dy);
                *(ofy++)  = bound_utils_y::sign(by, ny);
                *(oiy++)  = bound_utils_y::index(by, ny);
            }
        }
        {
            reduce_t    *owx = wx;
            offset_t    *oix = ix;
            signed char *ofx = fx;
            for (offset_t bx = bx0; bx <= bx1; ++bx) {
                scalar_t dx = x - bx;
                *(owx++)  = spline_utils_x::fastweight(dx);
                *(ofx++)  = bound_utils_x::sign(bx, nx);
                *(oix++)  = bound_utils_x::index(bx, nx);
            }
        }

        // Convolve coefficients with basis functions
        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t j = 0; j <= dby; ++j) {
            offset_t    oyy = iy[j] * sy;
            signed char fyy = fy[j];
            reduce_t    wyy = wy[j];
            for (offset_t i = 0; i <= dbx; ++i) {
                offset_t    oxy = oyy + ix[i] * sx;
                signed char fxy = fyy * fx[i];
                reduce_t    wxy = wyy * wx[i];
                acc += static_cast<reduce_t>(bound::get(inp, oxy, fxy)) * wxy;
            }
        }
        *out = static_cast<scalar_t>(acc);
    }
};

/***********************************************************************
 *
 *                                  3D
 *
 **********************************************************************/

/***                           NEAREST                              ***/
template <bound::type BX, bound::type BY, bound::type BZ>
struct PushPull<two, Z, BX, Z, BY, Z, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils = spline::utils<Z>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nx, offset_t sx, reduce_t wscl,
                offset_t h, offset_t ny, offset_t sy, reduce_t hscl,
                offset_t d, offset_t nz, offset_t sz, reduce_t dscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        reduce_t z = (d + shift) * dscl - shift;
        offset_t ix = static_cast<offset_t>(round(x));
        offset_t iy = static_cast<offset_t>(round(y));
        offset_t iz = static_cast<offset_t>(round(z));
        signed char  fx = bound_utils_x::sign(ix, nx);
        signed char  fy = bound_utils_y::sign(iy, ny);
        signed char  fz = bound_utils_z::sign(iz, nz);
        ix = bound_utils_x::index(ix, nx) * sx;
        iy = bound_utils_y::index(iy, ny) * sy;
        iz = bound_utils_z::index(iz, nz) * sz;

        *out = bound::get(inp, ix + iy + iz, fx * fy * fz);
    }
};

/***                            LINEAR                              ***/
template <bound::type BX, bound::type BY, bound::type BZ>
struct PushPull<three, L, BX, L, BY, L, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils = spline::utils<L>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nx, offset_t sx, reduce_t wscl,
                offset_t h, offset_t ny, offset_t sy, reduce_t hscl,
                offset_t d, offset_t nz, offset_t sz, reduce_t dscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        reduce_t z = (d + shift) * dscl - shift;
        offset_t ix0 = static_cast<offset_t>(floor(x));
        offset_t iy0 = static_cast<offset_t>(floor(y));
        offset_t iz0 = static_cast<offset_t>(floor(z));
        offset_t ix1 = ix0 + 1;
        offset_t iy1 = iy0 + 1;
        offset_t iz1 = iz0 + 1;
        reduce_t dx1 = x - ix0;
        reduce_t dy1 = y - iy0;
        reduce_t dz1 = z - iz0;
        reduce_t dx0 = 1 - dx1;
        reduce_t dy0 = 1 - dy1;
        reduce_t dz0 = 1 - dz1;
        signed char  fx0 = bound_utils_x::sign(ix0, nx);
        signed char  fy0 = bound_utils_y::sign(iy0, ny);
        signed char  fz0 = bound_utils_z::sign(iz0, nz);
        signed char  fx1 = bound_utils_x::sign(ix1, nx);
        signed char  fy1 = bound_utils_y::sign(iy1, ny);
        signed char  fz1 = bound_utils_z::sign(iz1, nz);
        ix0 = bound_utils_x::index(ix0, nx) * sx;
        iy0 = bound_utils_y::index(iy0, ny) * sy;
        iz0 = bound_utils_z::index(iz0, nz) * sz;
        ix1 = bound_utils_x::index(ix1, nx) * sx;
        iy1 = bound_utils_y::index(iy1, ny) * sy;
        iz1 = bound_utils_z::index(iz1, nz) * sz;

        auto accum1d = [ix0, ix1, dx0, dx1, fx0, fx1, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * fx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * fx1)) * dx1;
        };

        auto accum2d = [iy0, iy1, dy0, dy1, fy0, fy1, accum1d]
                        (offset_t i, signed char s)
        {
          return accum1d(iy0 + i, fy0 * s) * dy0
               + accum1d(iy1 + i, fy1 * s) * dy1;
        };

        *out  = static_cast<scalar_t>(accum2d(iz0, fz0) * dz0
                                    + accum2d(iz1, fz1) * dz1);
    }
};

/***                          QUADRATIC                             ***/
template <bound::type BX, bound::type BY, bound::type BZ>
struct PushPull<three, Q, BX, Q, BY, Q, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils = spline::utils<Q>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nx, offset_t sx, reduce_t wscl,
                offset_t h, offset_t ny, offset_t sy, reduce_t hscl,
                offset_t d, offset_t nz, offset_t sz, reduce_t dscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        reduce_t z = (d + shift) * dscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x+0.5));
        offset_t iy1 = static_cast<offset_t>(floor(y+0.5));
        offset_t iz1 = static_cast<offset_t>(floor(z+0.5));
        reduce_t dx1 = spline_utils::weight(x - ix1);
        reduce_t dy1 = spline_utils::weight(y - iy1);
        reduce_t dz1 = spline_utils::weight(z - iz1);
        reduce_t dx0 = spline_utils::fastweight(x - (ix1 - 1));
        reduce_t dy0 = spline_utils::fastweight(y - (iy1 - 1));
        reduce_t dz0 = spline_utils::fastweight(z - (iz1 - 1));
        reduce_t dx2 = spline_utils::fastweight((ix1 + 1) - x);
        reduce_t dy2 = spline_utils::fastweight((iy1 + 1) - y);
        reduce_t dz2 = spline_utils::fastweight((iz1 + 1) - z);
        signed char  fx0 = bound_utils_x::sign(ix1-1, nx);
        signed char  fy0 = bound_utils_y::sign(iy1-1, ny);
        signed char  fz0 = bound_utils_z::sign(iz1-1, nz);
        signed char  fx2 = bound_utils_x::sign(ix1+1, nx);
        signed char  fy2 = bound_utils_y::sign(iy1+1, ny);
        signed char  fz2 = bound_utils_z::sign(iz1+1, nz);
        signed char  fx1 = bound_utils_x::sign(ix1,   nx);
        signed char  fy1 = bound_utils_y::sign(iy1,   ny);
        signed char  fz1 = bound_utils_z::sign(iz1,   nz);
        offset_t ix0, iy0, iz0, ix2, iy2, iz2;
        ix0 = bound_utils_x::index(ix1-1, nx) * sx;
        iy0 = bound_utils_y::index(iy1-1, ny) * sy;
        iz0 = bound_utils_z::index(iz1-1, nz) * sz;
        ix2 = bound_utils_x::index(ix1+1, nx) * sx;
        iy2 = bound_utils_y::index(iy1+1, ny) * sy;
        iz2 = bound_utils_z::index(iz1+1, nz) * sz;
        ix1 = bound_utils_x::index(ix1,   nx) * sx;
        iy1 = bound_utils_y::index(iy1,   ny) * sy;
        iz1 = bound_utils_z::index(iz1,   nz) * sz;

        auto accum1d = [ix0, ix1, ix2, dx0, dx1, dx2, fx0, fx1, fx2, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * fx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * fx1)) * dx1
               + static_cast<reduce_t>(bound::get(inp, i + ix2, s * fx2)) * dx2;
        };

        auto accum2d = [iy0, iy1, iy2, dy0, dy1, dy2, fy0, fy1, fy2, accum1d]
                        (offset_t i, signed char s)
        {
          return accum1d(iy0 + i, fy0 * s) * dy0
               + accum1d(iy1 + i, fy1 * s) * dy1
               + accum1d(iy2 + i, fy2 * s) * dy2;
        };

        *out  = static_cast<scalar_t>(accum2d(iz0, fz0) * dz0
                                    + accum2d(iz1, fz1) * dz1
                                    + accum2d(iz2, fz2) * dz2);
    }
};

/***                            CUBIC                               ***/
template <bound::type BX, bound::type BY, bound::type BZ>
struct PushPull<three, C, BX, C, BY, C, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils = spline::utils<C>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nx, offset_t sx, reduce_t wscl,
                offset_t h, offset_t ny, offset_t sy, reduce_t hscl,
                offset_t d, offset_t nz, offset_t sz, reduce_t dscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        reduce_t z = (d + shift) * dscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x));
        offset_t iy1 = static_cast<offset_t>(floor(y));
        offset_t iz1 = static_cast<offset_t>(floor(z));
        reduce_t dx1 = spline_utils::fastweight(x - ix1);
        reduce_t dy1 = spline_utils::fastweight(y - iy1);
        reduce_t dz1 = spline_utils::fastweight(z - iz1);
        reduce_t dx0 = spline_utils::fastweight(x - (ix1 - 1));
        reduce_t dy0 = spline_utils::fastweight(y - (iy1 - 1));
        reduce_t dz0 = spline_utils::fastweight(z - (iz1 - 1));
        reduce_t dx2 = spline_utils::fastweight((ix1 + 1) - x);
        reduce_t dy2 = spline_utils::fastweight((iy1 + 1) - y);
        reduce_t dz2 = spline_utils::fastweight((iz1 + 1) - z);
        reduce_t dx3 = spline_utils::fastweight((ix1 + 2) - x);
        reduce_t dy3 = spline_utils::fastweight((iy1 + 2) - y);
        reduce_t dz3 = spline_utils::fastweight((iz1 + 2) - z);
        signed char  fx0 = bound_utils_x::sign(ix1-1, nx);
        signed char  fy0 = bound_utils_y::sign(iy1-1, ny);
        signed char  fz0 = bound_utils_z::sign(iz1-1, nz);
        signed char  fx2 = bound_utils_x::sign(ix1+1, nx);
        signed char  fy2 = bound_utils_y::sign(iy1+1, ny);
        signed char  fz2 = bound_utils_z::sign(iz1+1, nz);
        signed char  fx3 = bound_utils_x::sign(ix1+2, nx);
        signed char  fy3 = bound_utils_y::sign(iy1+2, ny);
        signed char  fz3 = bound_utils_z::sign(iz1+2, nz);
        signed char  fx1 = bound_utils_x::sign(ix1,   nx);
        signed char  fy1 = bound_utils_y::sign(iy1,   ny);
        signed char  fz1 = bound_utils_z::sign(iz1,   nz);
        offset_t ix0, ix2, ix3, iy0, iy2, iy3, iz0, iz2, iz3;
        ix0 = bound_utils_x::index(ix1-1, nx) * sx;
        iy0 = bound_utils_y::index(iy1-1, ny) * sy;
        iz0 = bound_utils_z::index(iz1-1, nz) * sz;
        ix2 = bound_utils_x::index(ix1+1, nx) * sx;
        iy2 = bound_utils_y::index(iy1+1, ny) * sy;
        iz2 = bound_utils_z::index(iz1+1, nz) * sz;
        ix3 = bound_utils_x::index(ix1+2, nx) * sx;
        iy3 = bound_utils_y::index(iy1+2, ny) * sy;
        iz3 = bound_utils_z::index(iz1+2, nz) * sz;
        ix1 = bound_utils_x::index(ix1,   nx) * sx;
        iy1 = bound_utils_y::index(iy1,   ny) * sy;
        iz1 = bound_utils_z::index(iz1,   nz) * sz;

        auto accum1d = [ix0, ix1, ix2, ix3, dx0, dx1, dx2, dx3,
                        fx0, fx1, fx2, fx3, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * fx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * fx1)) * dx1
               + static_cast<reduce_t>(bound::get(inp, i + ix2, s * fx2)) * dx2
               + static_cast<reduce_t>(bound::get(inp, i + ix3, s * fx3)) * dx3;
        };

        auto accum2d = [iy0, iy1, iy2, iy3, dy0, dy1, dy2, dy3,
                        fy0, fy1, fy2, fy3, accum1d]
                        (offset_t i, signed char s)
        {
          return accum1d(iy0 + i, fy0 * s) * dy0
               + accum1d(iy1 + i, fy1 * s) * dy1
               + accum1d(iy2 + i, fy2 * s) * dy2
               + accum1d(iy3 + i, fy3 * s) * dy3;
        };

        *out = static_cast<scalar_t>(accum2d(iz0, fz0) * dz0
                                   + accum2d(iz1, fz1) * dz1
                                   + accum2d(iz2, fz2) * dz2
                                   + accum2d(iz3, fz3) * dz3);
    }
};

/***                             ANY                                ***/
template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ>
struct PushPull<three, IX, BX, IY, BY, IZ, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils_x = spline::utils<IX>;
    using spline_utils_y = spline::utils<IY>;
    using spline_utils_z = spline::utils<IZ>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nx, offset_t sx, reduce_t wscl,
                offset_t h, offset_t ny, offset_t sy, reduce_t hscl,
                offset_t d, offset_t nz, offset_t sz, reduce_t dscl,
                reduce_t shift)
    {
        // Precompute weights and indices
        reduce_t x = wscl * (w + shift) - shift;
        reduce_t y = hscl * (h + shift) - shift;
        reduce_t z = dscl * (d + shift) - shift;
        offset_t bx0, bx1, by0, by1, bz0, bz1;
        spline_utils_x::bounds(x, bx0, bx1);
        spline_utils_y::bounds(y, by0, by1);
        spline_utils_z::bounds(z, bz0, bz1);
        offset_t dbx = bx1-bx0;
        offset_t dby = by1-by0;
        offset_t dbz = bz1-bz0;
        reduce_t    wx[8],  wy[8],  wz[8];
        offset_t    ix[8],  iy[8],  iz[8];
        signed char fx[8],  fy[8],  fz[8];
        {
            reduce_t    *owz = wz;
            offset_t    *oiz = iz;
            signed char *ofz = fz;
            for (offset_t bz = bz0; bz <= bz1; ++bz) {
                scalar_t dz = z - bz;
                *(owz++)  = spline_utils_z::fastweight(dz);
                *(ofz++)  = bound_utils_z::sign(bz, nz);
                *(oiz++)  = bound_utils_z::index(bz, nz);
            }
        }
        {
            reduce_t    *owy = wy;
            offset_t    *oiy = iy;
            signed char *ofy = fy;
            for (offset_t by = by0; by <= by1; ++by) {
                scalar_t dy = y - by;
                *(owy++)  = spline_utils_y::fastweight(dy);
                *(ofy++)  = bound_utils_y::sign(by, ny);
                *(oiy++)  = bound_utils_y::index(by, ny);
            }
        }
        {
            reduce_t    *owx = wx;
            offset_t    *oix = ix;
            signed char *ofx = fx;
            for (offset_t bx = bx0; bx <= bx1; ++bx) {
                scalar_t dx = x - bx;
                *(owx++)  = spline_utils_x::fastweight(dx);
                *(ofx++)  = bound_utils_x::sign(bx, nx);
                *(oix++)  = bound_utils_x::index(bx, nx);
            }
        }

        // Convolve coefficients with basis functions
        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t k = 0; k <= dbz; ++k) {
            offset_t    ozz = iz[k] * sz;
            signed char fzz = fz[k];
            reduce_t    wzz = wz[k];
            for (offset_t j = 0; j <= dby; ++j) {
                offset_t    oyz = ozz + iy[j] * sy;
                signed char fyz = fzz * fy[j];
                reduce_t    wyz = wzz * wy[j];
                for (offset_t i = 0; i <= dbx; ++i) {
                    offset_t    oxyz = oyz + ix[i] * sx;
                    signed char fxyz = fyz * fx[i];
                    reduce_t    wxyz = wyz * wx[i];
                    acc += static_cast<reduce_t>(bound::get(inp, oxyz, fxyz)) * wxyz;
                }
            }
        }
        *out = static_cast<scalar_t>(acc);
    }
};

/***********************************************************************
 *
 *                                  ND
 *
 **********************************************************************/

template <int D>
struct PushPull<D> {

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                const offset_t * coord, const offset_t * size, const offset_t * stride,
                const spline::type * inter, const bound::type * bnd,
                const reduce_t * scl, reduce_t shift)
    {
        // Precompute weights and indices
        reduce_t    w[8*D];
        offset_t    i[8*D];
        signed char s[8*D];
        offset_t    db[D];
        for (int d=0; d<D; ++d) {
            reduce_t    *wd = w + 8*d;
            offset_t    *id = i + 8*d;
            signed char *sz = s + 8*d;
            reduce_t x = scl[d] * (coord[d] + shift) - shift;
            offset_t b0, b1;
            spline::bounds(inter[d], x, b0, b1);
            db[d] = b1-b0;
            for (offset_t b = b0; b <= b1; ++b) {
                *(wd++) = spline::fastweight(inter[d], x - b);
                *(sz++) = bound::sign(bnd[d], b, size[d]);
                *(id++) = bound::index(bnd[d], b, size[d]);
            }
        }

        // Convolve coefficients with basis functions
        offset_t    offsets[D];
        signed char signs[D];
        scalar_t    weights[D];
        reduce_t acc = static_cast<reduce_t>(0);
        for (int d=0; d<D; ++d) {
            reduce_t    *wd = w + 8*d;
            offset_t    *id = i + 8*d;
            signed char *sz = s + 8*d;
            for (offset_t k = 0; k <= db[d]; ++k) {
                offsets[d] = (d > 0 ? offsets[d-1] : static_cast<offset_t>(0))
                           + id[k] * stride[d];
                signs[d]   = (d > 0 ? signs[d-1]   : static_cast<signed char>(1))
                           * sz[k];
                weights[d] = (d > 0 ? weights[d-1] : static_cast<reduce_t>(1))
                           * wd[k];
                if (d == D-1)
                    acc += static_cast<reduce_t>(bound::get(inp, offsets[D-1], signs[D-1])) * weights[D-1];
            }
        }
        *out = static_cast<scalar_t>(acc);
    }
};

#endif

} // namespace resize
} // namespace jf

#endif // JF_PUSHPULL
