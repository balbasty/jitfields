/***********************************************************************
 *
 *                                  2D
 *
 **********************************************************************/
#ifndef JF_PUSHPULL_2D
#define JF_PUSHPULL_2D
#include "../cuda_switch.h"
#include "../spline.h"
#include "../bounds.h"
#include "utils.h"

// TODO: quadratic and cubic specializations

namespace jf {
namespace pushpull {

/***********************************************************************
 *
 *                               NEAREST
 *
 **********************************************************************/
template <bound::type BX, bound::type BY>
struct PushPull<two, Z, BX, Z, BY> {
    using utils_x = PushPullUtils<Z, BX>;
    using utils_y = PushPullUtils<Z, BY>;
    using self = PushPull<two, Z, BX, Z, BY>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[2],
              const offset_t size[2],
              const offset_t stride[2],
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix, iy;
        signed char fx, fy;
        utils_x::index(loc[0], size[0], ix, fx);
        utils_y::index(loc[1], size[1], iy, fy);
        offset_t    i = ix * stride[0] + iy * stride[1];
        signed char f = fx * fy;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = bound::get(inp, i, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[2],
              const offset_t size[2],
              const offset_t stride[2],
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix, iy;
        signed char fx, fy;
        utils_x::index(loc[0], size[0], ix, fx);
        utils_y::index(loc[1], size[1], iy, fy);
        offset_t    i = ix * stride[0] + iy * stride[1];
        signed char f = fx * fy;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            bound::add(out, i, *inp, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count(scalar_t * out,
              const reduce_t loc[2],
              const offset_t size[2],
              const offset_t stride[2])
    {
        offset_t    ix, iy;
        signed char fx, fy;
        utils_x::index(loc[0], size[0], ix, fx);
        utils_y::index(loc[1], size[1], iy, fy);
        offset_t    i = ix * stride[0] + iy * stride[1];
        signed char f = fx * fy;

        bound::add(out, i, 1, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[2],
              const offset_t size[2],
              const offset_t stride[2],
              offset_t nc, offset_t osc, offset_t isc, offset_t osg)
    {
        for (offset_t c = 0; c < nc; ++c, out += osc) {
            *out     = static_cast<scalar_t>(0);
            out[osg] = static_cast<scalar_t>(0);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[2],
                       const offset_t size[2],
                       const offset_t stride_out[2],
                       const offset_t stride_inp[2],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        gout[0]   = static_cast<scalar_t>(0);
        gout[osg] = static_cast<scalar_t>(0);
        self::push(out, ginp, loc, size, stride_out, nc, osc, isc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[2],
                       const offset_t size[2],
                       const offset_t stride[2],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        gout[0]   = static_cast<scalar_t>(0);
        gout[osg] = static_cast<scalar_t>(0);
        self::pull(out, ginp, loc, size, stride, nc, osc, isc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count_backward(scalar_t * gout, const scalar_t * inp,
                        const reduce_t loc[2],
                        const offset_t size[2],
                        const offset_t stride[2],
                        offset_t osg)
    {
        gout[0]   = static_cast<scalar_t>(0);
        gout[osg] = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[2],
                       const offset_t size[2],
                       const offset_t stride_out[2],
                       const offset_t stride_inp[2],
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc,
                       offset_t osg, offset_t isg)
    {
        gout[0]   = static_cast<scalar_t>(0);
        gout[osg] = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }
};


/***********************************************************************
 *
 *                               LINEAR
 *
 **********************************************************************/
template <bound::type BX, bound::type BY>
struct PushPull<two, L, BX, L, BY> {
    using utils_x = PushPullUtils<L, BX>;
    using utils_y = PushPullUtils<L, BY>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[2],
              const offset_t size[2],
              const offset_t stride[2],
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix0, ix1, iy0, iy1;
        reduce_t    wx0, wx1, wy0, wy1;
        signed char fx0, fx1, fy0, fy1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        reduce_t f00 = fx0 * fy0;
        reduce_t f01 = fx0 * fy1;
        reduce_t f10 = fx1 * fy0;
        reduce_t f11 = fx1 * fy1;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, i00, f00) * w00
                    + bound::cget<reduce_t>(inp, i01, f01) * w01
                    + bound::cget<reduce_t>(inp, i10, f10) * w10
                    + bound::cget<reduce_t>(inp, i11, f11) * w11);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[2],
              const offset_t size[2],
              const offset_t stride[2],
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix0, ix1, iy0, iy1;
        reduce_t    wx0, wx1, wy0, wy1;
        signed char fx0, fx1, fy0, fy1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        reduce_t f00 = fx0 * fy0;
        reduce_t f01 = fx0 * fy1;
        reduce_t f10 = fx1 * fy0;
        reduce_t f11 = fx1 * fy1;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, i00, val * w00, f00);
            bound::add(out, i01, val * w01, f01);
            bound::add(out, i10, val * w10, f10);
            bound::add(out, i11, val * w11, f11);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count(scalar_t * out,
              const reduce_t loc[2],
              const offset_t size[2],
              const offset_t stride[2])
    {
        offset_t    ix0, ix1, iy0, iy1;
        reduce_t    wx0, wx1, wy0, wy1;
        signed char fx0, fx1, fy0, fy1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        reduce_t f00 = fx0 * fy0;
        reduce_t f01 = fx0 * fy1;
        reduce_t f10 = fx1 * fy0;
        reduce_t f11 = fx1 * fy1;

        bound::add(out, i00, w00, f00);
        bound::add(out, i01, w01, f01);
        bound::add(out, i10, w10, f10);
        bound::add(out, i11, w11, f11);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[2],
              const offset_t size[2],
              const offset_t stride[2],
              offset_t nc, offset_t osc, offset_t isc,
              offset_t osg)
    {
        offset_t    ix0, ix1, iy0, iy1;
        reduce_t    wx0, wx1, wy0, wy1;
        signed char fx0, fx1, fy0, fy1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        reduce_t f00 = fx0 * fy0;
        reduce_t f01 = fx0 * fy1;
        reduce_t f10 = fx1 * fy0;
        reduce_t f11 = fx1 * fy1;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t v00 = bound::cget<reduce_t>(inp, i00, f00);
            reduce_t v01 = bound::cget<reduce_t>(inp, i01, f01);
            reduce_t v10 = bound::cget<reduce_t>(inp, i10, f10);
            reduce_t v11 = bound::cget<reduce_t>(inp, i11, f11);
            out[0] = static_cast<scalar_t>(
                    - v00 * wy0 - v01 * wy1 + v10 * wy0 + v11 * wy1);
            out[osg] = static_cast<scalar_t>(
                    - v00 * wx0 + v01 * wx0 - v10 * wx1 + v11 * wx1);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[2],
                       const offset_t size[2],
                       const offset_t stride_out[2],
                       const offset_t stride_inp[2],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        offset_t    ix0, ix1, iy0, iy1;
        reduce_t    wx0, wx1, wy0, wy1;
        signed char fx0, fx1, fy0, fy1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        offset_t osx = stride_out[0], osy = stride_out[1], osz = stride_out[2];
        offset_t isx = stride_inp[0], isy = stride_inp[1], isz = stride_inp[2];
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        reduce_t f00 = fx0 * fy0;
        reduce_t f01 = fx0 * fy1;
        reduce_t f10 = fx1 * fy0;
        reduce_t f11 = fx1 * fy1;
        offset_t i00 = ix0 * isx + iy0 * isy;
        offset_t i01 = ix0 * isx + iy1 * isy;
        offset_t i10 = ix1 * isx + iy0 * isy;
        offset_t i11 = ix1 * isx + iy1 * isy;
        offset_t o00 = ix0 * osx + iy0 * osy;
        offset_t o01 = ix0 * osx + iy1 * osy;
        offset_t o10 = ix1 * osx + iy0 * osy;
        offset_t o11 = ix1 * osx + iy1 * osy;

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, o00, gval * w00, f00);
            bound::add(out, o01, gval * w01, f01);
            bound::add(out, o10, gval * w10, f10);
            bound::add(out, o11, gval * w11, f11);
            // compute input spatial gradient
            reduce_t v00 = bound::cget<reduce_t>(inp, i00, f00);
            reduce_t v01 = bound::cget<reduce_t>(inp, i01, f01);
            reduce_t v10 = bound::cget<reduce_t>(inp, i10, f10);
            reduce_t v11 = bound::cget<reduce_t>(inp, i11, f11);
            accx += gval * (- v00 * wy0 - v01 * wy1 + v10 * wy0 + v11 * wy1);
            accy += gval * (- v00 * wx0 + v01 * wx0 - v10 * wx1 + v11 * wx1);
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[2],
                       const offset_t size[2],
                       const offset_t stride[2],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        offset_t    ix0, ix1, iy0, iy1;
        reduce_t    wx0, wx1, wy0, wy1;
        signed char fx0, fx1, fy0, fy1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];
        offset_t i00 = ix0 * sx + iy0 * sy;
        offset_t i01 = ix0 * sx + iy1 * sy;
        offset_t i10 = ix1 * sx + iy0 * sy;
        offset_t i11 = ix1 * sx + iy1 * sy;
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        reduce_t f00 = fx0 * fy0;
        reduce_t f01 = fx0 * fy1;
        reduce_t f10 = fx1 * fy0;
        reduce_t f11 = fx1 * fy1;

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            // pull incoming gradient
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(ginp, i00, f00) * w00
                    + bound::cget<reduce_t>(ginp, i01, f01) * w01
                    + bound::cget<reduce_t>(ginp, i10, f10) * w10
                    + bound::cget<reduce_t>(ginp, i11, f11) * w11);
            // compute input spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            reduce_t v00 = bound::cget<reduce_t>(ginp, i00, f00);
            reduce_t v01 = bound::cget<reduce_t>(ginp, i01, f01);
            reduce_t v10 = bound::cget<reduce_t>(ginp, i10, f10);
            reduce_t v11 = bound::cget<reduce_t>(ginp, i11, f11);
            accx += val * (- v00 * wy0 - v01 * wy1 + v10 * wy0 + v11 * wy1);
            accy += val * (- v00 * wx0 + v01 * wx0 - v10 * wx1 + v11 * wx1);
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count_backward(scalar_t * gout, const scalar_t * ginp,
                        const reduce_t loc[2],
                        const offset_t size[2],
                        const offset_t stride[2],
                        offset_t osg)
    {
        offset_t    ix0, ix1, iy0, iy1;
        reduce_t    wx0, wx1, wy0, wy1;
        signed char fx0, fx1, fy0, fy1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];
        offset_t i00 = ix0 * sx + iy0 * sy;
        offset_t i01 = ix0 * sx + iy1 * sy;
        offset_t i10 = ix1 * sx + iy0 * sy;
        offset_t i11 = ix1 * sx + iy1 * sy;
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        reduce_t f00 = fx0 * fy0;
        reduce_t f01 = fx0 * fy1;
        reduce_t f10 = fx1 * fy0;
        reduce_t f11 = fx1 * fy1;

        // compute input spatial gradient
        reduce_t v00 = bound::cget<reduce_t>(ginp, i00, f00);
        reduce_t v01 = bound::cget<reduce_t>(ginp, i01, f01);
        reduce_t v10 = bound::cget<reduce_t>(ginp, i10, f10);
        reduce_t v11 = bound::cget<reduce_t>(ginp, i11, f11);
        gout[0]   = static_cast<scalar_t>(
                  - v00 * wy0 - v01 * wy1 + v10 * wy0 + v11 * wy1);
        gout[osg] = static_cast<scalar_t>(
                  - v00 * wx0 + v01 * wx0 - v10 * wx1 + v11 * wx1);
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[2],
                       const offset_t size[2],
                       const offset_t stride_out[2],
                       const offset_t stride_inp[2],
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc,
                       offset_t osg, offset_t isg)
    {
        offset_t    ix0, ix1, iy0, iy1;
        reduce_t    wx0, wx1, wy0, wy1;
        signed char fx0, fx1, fy0, fy1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        offset_t osx = stride_out[0], osy = stride_out[1];
        offset_t isx = stride_inp[0], isy = stride_inp[1];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t oval;
            reduce_t gvalx = static_cast<reduce_t>(ginp[0]);
            reduce_t gvaly = static_cast<reduce_t>(ginp[isg]);

            oval = - gvalx * wy0 - gvaly * wx0;
            bound::add(out, ix0 * osx + iy0 * osy, oval, fx0 * fy0);

            oval = - gvalx * wy1 + gvaly * wx0;
            bound::add(out, ix0 * osx + iy1 * osy, oval, fx0 * fy1);

            oval = + gvalx * wy0 - gvaly * wx1;
            bound::add(out, ix1 * osx + iy0 * osy, oval, fx1 * fy0);

            oval = + gvalx * wy1 + gvaly * wx1;
            bound::add(out, ix1 * osx + iy1 * osy, oval, fx1 * fy1);
        }

        gout[0]   = static_cast<scalar_t>(0);
        gout[osg] = static_cast<scalar_t>(0);
    }
};


/***********************************************************************
 *
 *                                 ANY
 *
 **********************************************************************/
template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY>
struct PushPull<two, IX, BX, IY, BY> {
    using utils_x = PushPullAnyUtils<IX, BX>;
    using utils_y = PushPullAnyUtils<IY, BY>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[2],
              const offset_t size[2],
              const offset_t stride[2],
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8];
        reduce_t    wx[8], wy[8];
        signed char fx[8], fy[8];
        offset_t lx = utils_x::index(loc[0], size[1], ix, wx, fx);
        offset_t ly = utils_y::index(loc[1], size[1], iy, wy, fy);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= lx; ++i)
            for (offset_t j = 0; j <= ly; ++j)
                acc += bound::cget<reduce_t>(
                    inp, ix[i] + iy[j], fx[i] * fy[j]) * (wx[i] * wy[j]);
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[2],
              const offset_t size[2],
              const offset_t stride[2],
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8];
        reduce_t    wx[8], wy[8];
        signed char fx[8], fy[8];
        offset_t lx = utils_x::index(loc[0], size[0], ix, wx, fx);
        offset_t ly = utils_y::index(loc[1], size[1], iy, wy, fy);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            for (offset_t i = 0; i <= lx; ++i)
            for (offset_t j = 0; j <= ly; ++j)
                bound::add(out, ix[i] + iy[j], val * (wx[i] * wy[j]), fx[i] * fy[j]);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count(scalar_t * out,
               const reduce_t loc[2],
               const offset_t size[2],
               const offset_t stride[2])
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8];
        reduce_t    wx[8], wy[8];
        signed char fx[8], fy[8];
        offset_t lx = utils_x::index(loc[0], size[0], ix, wx, fx);
        offset_t ly = utils_y::index(loc[1], size[1], iy, wy, fy);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;

        for (offset_t i = 0; i <= lx; ++i)
        for (offset_t j = 0; j <= ly; ++j)
            bound::add(out, ix[i] + iy[j], wx[i] * wy[j], fx[i] * fy[j]);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[2],
              const offset_t size[2],
              const offset_t stride[2],
              offset_t nc, offset_t osc, offset_t isc, offset_t osg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8];
        reduce_t    wx[8], wy[8];
        reduce_t    gx[8], gy[8];
        signed char fx[8], fy[8];
        offset_t lx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx);
        offset_t ly = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t accx = static_cast<reduce_t>(0);
            reduce_t accy = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= lx; ++i)
            for (offset_t j = 0; j <= ly; ++j) {
                reduce_t val = bound::cget<reduce_t>(inp, ix[i] + iy[j], fx[i] * fy[j]);
                accx += val * (gx[i] * wy[j]);
                accy += val * (wx[i] * gy[j]);
            }
            out[0]   = static_cast<scalar_t>(accx);
            out[osg] = static_cast<scalar_t>(accy);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[2],
                       const offset_t size[2],
                       const offset_t stride_out[2],
                       const offset_t stride_inp[2],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8];
        reduce_t    wx[8], wy[8];
        reduce_t    gx[8], gy[8];
        signed char fx[8], fy[8];
        offset_t lx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx);
        offset_t ly = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy);
        offset_t osx = stride_out[0], osy = stride_out[1];
        offset_t isx = stride_inp[0], isy = stride_inp[1];

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            reduce_t gval  = static_cast<reduce_t>(*ginp);
            reduce_t accx1 = static_cast<reduce_t>(0);
            reduce_t accy1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= lx; ++i) {
                offset_t    ixo = ix[i] * osx;
                offset_t    ixi = ix[i] * isx;
                signed char ffx  = fx[i];
                reduce_t    wwx  = wx[i];
                reduce_t    ggx  = gx[i];
                for (offset_t j = 0; j <= ly; ++j) {
                    offset_t    iyo = ixo + iy[j] * osy;
                    offset_t    iyi = ixi + iy[j] * isy;
                    signed char ff = ffx * fy[j];
                    // push incoming gradient
                    bound::add(out, iyo, gval * (wwx * wy[j]), ff);
                    // compute input spatial gradient
                    reduce_t val = bound::cget<reduce_t>(inp, iyi, ff);
                    accx1 += val * (ggx * wy[j]);
                    accy1 += val * (wwx * gy[j]);
                }
            }
            accx += gval * accx1;
            accy += gval * accy1;
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[2],
                       const offset_t size[2],
                       const offset_t stride[2],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8];
        reduce_t    wx[8], wy[8];
        reduce_t    gx[8], gy[8];
        signed char fx[8], fy[8];
        offset_t lx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx);
        offset_t ly = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            reduce_t val = static_cast<reduce_t>(*inp);
            reduce_t acc1 = static_cast<reduce_t>(0);
            reduce_t accx2 = static_cast<reduce_t>(0);
            reduce_t accy2 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= lx; ++i)
                for (offset_t j = 0; j <= ly; ++j) {
                    reduce_t gval = bound::cget<reduce_t>(ginp, ix[i] + iy[j], fx[i] * fy[j]);
                    // pull incoming gradient
                    acc1 += gval * (wx[i] * wy[j]);
                    // compute incoming gradient spatial gradient
                    accx2 += gval * (gx[i] * wy[j]);
                    accy2 += gval * (wx[i] * gy[j]);
            }
            *out = static_cast<scalar_t>(acc1);
            accx += val * accx2;
            accy += val * accy2;
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count_backward(scalar_t * gout, const scalar_t * ginp,
                        const reduce_t loc[2],
                        const offset_t size[2],
                        const offset_t stride[2],
                        offset_t osg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8];
        reduce_t    wx[8], wy[8];
        reduce_t    gx[8], gy[8];
        signed char fx[8], fy[8];
        offset_t lx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx);
        offset_t ly = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;

        // compute input spatial gradient
        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t i = 0; i <= lx; ++i)
        for (offset_t j = 0; j <= ly; ++j) {
            reduce_t val = bound::cget<reduce_t>(ginp, ix[i] + iy[j], fx[i] * fy[j]);
            accx += val * (gx[i] * wy[j]);
            accy += val * (wx[i] * gy[j]);
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                        const reduce_t loc[2],
                        const offset_t size[2],
                        const offset_t stride_out[2],
                        const offset_t stride_inp[2],
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc,
                       offset_t osg, offset_t isg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8];
        reduce_t    wx[8], wy[8];
        reduce_t    gx[8], gy[8];
        reduce_t    hx[8], hy[8];
        signed char fx[8], fy[8];
        offset_t lx = utils_x::hindex(loc[0], size[0], ix, wx, gx, hx, fx);
        offset_t ly = utils_y::hindex(loc[1], size[1], iy, wy, gy, hy, fy);
        offset_t osx = stride_out[0], osy = stride_out[1];
        offset_t isx = stride_inp[0], isy = stride_inp[1];

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t gvalx = static_cast<reduce_t>(ginp[0]);
            reduce_t gvaly = static_cast<reduce_t>(ginp[isg]);
            reduce_t accxx1 = static_cast<reduce_t>(0);
            reduce_t accxy1 = static_cast<reduce_t>(0);
            reduce_t accyy1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= lx; ++i)
            for (offset_t j = 0; j <= ly; ++j) {
                // push incoming gradient
                reduce_t oval = gvalx * (gx[i] * wy[j]) + gvaly * (wx[i] * gy[j]);
                bound::add(out, ix[i] * osx + iy[j] * osy, oval, fx[i] * fy[j]);
                // compute input spatial hessian
                reduce_t ival = bound::cget<reduce_t>(inp, ix[i] * isx + iy[j] * isy, fx[i] * fy[j]);
                accxx1 += ival * hx[i] * wy[j];
                accyy1 += ival * wx[i] * hy[j];
                accxy1 += ival * gx[i] * gy[j];
            }
            accx += gvalx * accxx1 + gvaly * accxy1;
            accy += gvaly * accyy1 + gvalx * accxy1;
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }
};

} // namespace pushpull
} // namespace jf

#endif // JF_PUSHPULL_2D
