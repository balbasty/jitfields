/***********************************************************************
 *
 *                                  3D
 *
 **********************************************************************/
#ifndef JF_PUSHPULL_3D
#define JF_PUSHPULL_3D
#include "cuda_switch.h"
#include "spline.h"
#include "bounds.h"
#include "pushpull_utils.h"

// TODO: quadratic and cubic specializations

namespace jf {
namespace pushpull {

/***********************************************************************
 *
 *                               NEAREST
 *
 **********************************************************************/
template <bound::type BX, bound::type BY, bound::type BZ>
struct PushPull<three, Z, BX, Z, BY, Z, BZ> {
    using utils_x = PushPullUtils<Z, BX>;
    using utils_y = PushPullUtils<Z, BY>;
    using utils_z = PushPullUtils<Z, BZ>;
    using self = PushPull<three, Z, BX, Z, BY, Z, BZ>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[3],
              const offset_t size[3],
              const offset_t stride[3],
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix, iy, iz;
        signed char fx, fy, fz;
        utils_x::index(loc[0], size[0], ix, fx);
        utils_y::index(loc[1], size[1], iy, fy);
        utils_z::index(loc[2], size[2], iz, fz);
        offset_t    i = ix * stride[0] + iy * stride[1] + iz * stride[2];
        signed char f = fx * fy * fz;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = bound::get(inp, i, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[3],
              const offset_t size[3],
              const offset_t stride[3],
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix, iy, iz;
        signed char fx, fy, fz;
        utils_x::index(loc[0], size[0], ix, fx);
        utils_y::index(loc[1], size[1], iy, fy);
        utils_z::index(loc[2], size[2], iz, fz);
        offset_t    i = ix * stride[0] + iy * stride[1] + iz * stride[2];
        signed char f = fx * fy * fz;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            bound::add(out, i, *inp, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count(scalar_t * out,
              const reduce_t loc[3],
              const offset_t size[3],
              const offset_t stride[3])
    {
        offset_t    ix, iy, iz;
        signed char fx, fy, fz;
        utils_x::index(loc[0], size[0], ix, fx);
        utils_y::index(loc[1], size[1], iy, fy);
        utils_z::index(loc[2], size[2], iz, fz);
        offset_t    i = ix * stride[0] + iy * stride[1] + iz * stride[2];
        signed char f = fx * fy * fz;

        bound::add(out, i, 1, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[3],
              const offset_t size[3],
              const offset_t stride[3],
              offset_t nc, offset_t osc, offset_t isc, offset_t osg)
    {
        for (offset_t c = 0; c < nc; ++c, out += osc) {
            out[0]       = static_cast<scalar_t>(0);
            out[osg]     = static_cast<scalar_t>(0);
            out[osg * 2] = static_cast<scalar_t>(0);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[3],
                       const offset_t size[3],
                       const offset_t stride_out[3],
                       const offset_t stride_inp[3],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        gout[0]       = static_cast<scalar_t>(0);
        gout[osg]     = static_cast<scalar_t>(0);
        gout[osg * 2] = static_cast<scalar_t>(0);
        self::push(out, ginp, loc, size, stride_out, nc, osc, isc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[3],
                       const offset_t size[3],
                       const offset_t stride[3],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        gout[0]       = static_cast<scalar_t>(0);
        gout[osg]     = static_cast<scalar_t>(0);
        gout[osg * 2] = static_cast<scalar_t>(0);
        self::pull(out, ginp, loc, size, stride, nc, osc, isc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count_backward(scalar_t * gout, const scalar_t * inp,
                        const reduce_t loc[3],
                        const offset_t size[3],
                        const offset_t stride[3],
                        offset_t osg)
    {
        gout[0]       = static_cast<scalar_t>(0);
        gout[osg]     = static_cast<scalar_t>(0);
        gout[osg * 2] = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[3],
                       const offset_t size[3],
                       const offset_t stride_out[3],
                       const offset_t stride_inp[3],
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc,
                       offset_t osg, offset_t isg)
    {
        gout[0]       = static_cast<scalar_t>(0);
        gout[osg]     = static_cast<scalar_t>(0);
        gout[osg * 2] = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }
};


/***********************************************************************
 *
 *                               LINEAR
 *
 **********************************************************************/
template <bound::type BX, bound::type BY, bound::type BZ>
struct PushPull<three, L, BX, L, BY, L, BZ> {
    using utils_x = PushPullUtils<L, BX>;
    using utils_y = PushPullUtils<L, BY>;
    using utils_z = PushPullUtils<L, BZ>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[3],
              const offset_t size[3],
              const offset_t stride[3],
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix0, ix1, iy0, iy1, iz0, iz1;
        reduce_t    wx0, wx1, wy0, wy1, wz0, wz1;
        signed char fx0, fx1, fy0, fy1, fz0, fz1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        utils_z::index(loc[2], size[2], iz0, iz1, wz0, wz1, fz0, fz1);
        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        iz0 *= stride[2]; iz1 *= stride[2];
        offset_t i000, i001, i010, i011, i100, i101, i110, i111;
        {
            offset_t i00 = iy0 + iz0;
            offset_t i01 = iy0 + iz1;
            offset_t i10 = iy1 + iz0;
            offset_t i11 = iy1 + iz1;
            i000 = ix0 + i00;
            i001 = ix0 + i01;
            i010 = ix0 + i10;
            i011 = ix0 + i11;
            i100 = ix1 + i00;
            i101 = ix1 + i01;
            i110 = ix1 + i10;
            i111 = ix1 + i11;
        }
        reduce_t w000, w001, w010, w011, w100, w101, w110, w111;
        {
            reduce_t w00 = wy0 * wz0;
            reduce_t w01 = wy0 * wz1;
            reduce_t w10 = wy1 * wz0;
            reduce_t w11 = wy1 * wz1;
            w000 = wx0 * w00;
            w001 = wx0 * w01;
            w010 = wx0 * w10;
            w011 = wx0 * w11;
            w100 = wx1 * w00;
            w101 = wx1 * w01;
            w110 = wx1 * w10;
            w111 = wx1 * w11;
        }
        signed char f000, f001, f010, f011, f100, f101, f110, f111;
        {
            reduce_t f00 = fy0 * fz0;
            reduce_t f01 = fy0 * fz1;
            reduce_t f10 = fy1 * fz0;
            reduce_t f11 = fy1 * fz1;
            f000 = fx0 * f00;
            f001 = fx0 * f01;
            f010 = fx0 * f10;
            f011 = fx0 * f11;
            f100 = fx1 * f00;
            f101 = fx1 * f01;
            f110 = fx1 * f10;
            f111 = fx1 * f11;
        }

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, i000, f000) * w000
                    + bound::cget<reduce_t>(inp, i001, f001) * w001
                    + bound::cget<reduce_t>(inp, i010, f010) * w010
                    + bound::cget<reduce_t>(inp, i011, f011) * w011
                    + bound::cget<reduce_t>(inp, i100, f100) * w100
                    + bound::cget<reduce_t>(inp, i101, f101) * w101
                    + bound::cget<reduce_t>(inp, i110, f110) * w110
                    + bound::cget<reduce_t>(inp, i111, f111) * w111);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[3],
              const offset_t size[3],
              const offset_t stride[3],
              offset_t nc, offset_t osc, offset_t isc)
    {
        offset_t    ix0, ix1, iy0, iy1, iz0, iz1;
        reduce_t    wx0, wx1, wy0, wy1, wz0, wz1;
        signed char fx0, fx1, fy0, fy1, fz0, fz1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        utils_z::index(loc[2], size[2], iz0, iz1, wz0, wz1, fz0, fz1);
        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        iz0 *= stride[2]; iz1 *= stride[2];
        offset_t i000, i001, i010, i011, i100, i101, i110, i111;
        {
            offset_t i00 = iy0 + iz0;
            offset_t i01 = iy0 + iz1;
            offset_t i10 = iy1 + iz0;
            offset_t i11 = iy1 + iz1;
            i000 = ix0 + i00;
            i001 = ix0 + i01;
            i010 = ix0 + i10;
            i011 = ix0 + i11;
            i100 = ix1 + i00;
            i101 = ix1 + i01;
            i110 = ix1 + i10;
            i111 = ix1 + i11;
        }
        reduce_t w000, w001, w010, w011, w100, w101, w110, w111;
        {
            reduce_t w00 = wy0 * wz0;
            reduce_t w01 = wy0 * wz1;
            reduce_t w10 = wy1 * wz0;
            reduce_t w11 = wy1 * wz1;
            w000 = wx0 * w00;
            w001 = wx0 * w01;
            w010 = wx0 * w10;
            w011 = wx0 * w11;
            w100 = wx1 * w00;
            w101 = wx1 * w01;
            w110 = wx1 * w10;
            w111 = wx1 * w11;
        }
        signed char f000, f001, f010, f011, f100, f101, f110, f111;
        {
            reduce_t f00 = fy0 * fz0;
            reduce_t f01 = fy0 * fz1;
            reduce_t f10 = fy1 * fz0;
            reduce_t f11 = fy1 * fz1;
            f000 = fx0 * f00;
            f001 = fx0 * f01;
            f010 = fx0 * f10;
            f011 = fx0 * f11;
            f100 = fx1 * f00;
            f101 = fx1 * f01;
            f110 = fx1 * f10;
            f111 = fx1 * f11;
        }

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, i000, val * w000, f000);
            bound::add(out, i001, val * w001, f001);
            bound::add(out, i010, val * w010, f010);
            bound::add(out, i011, val * w011, f011);
            bound::add(out, i100, val * w100, f100);
            bound::add(out, i101, val * w101, f101);
            bound::add(out, i110, val * w110, f110);
            bound::add(out, i111, val * w111, f111);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count(scalar_t * out,
               const reduce_t loc[3],
               const offset_t size[3],
               const offset_t stride[3])
    {
        offset_t    ix0, ix1, iy0, iy1, iz0, iz1;
        reduce_t    wx0, wx1, wy0, wy1, wz0, wz1;
        signed char fx0, fx1, fy0, fy1, fz0, fz1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        utils_z::index(loc[2], size[2], iz0, iz1, wz0, wz1, fz0, fz1);
        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        iz0 *= stride[2]; iz1 *= stride[2];
        offset_t i000, i001, i010, i011, i100, i101, i110, i111;
        {
            offset_t i00 = iy0 + iz0;
            offset_t i01 = iy0 + iz1;
            offset_t i10 = iy1 + iz0;
            offset_t i11 = iy1 + iz1;
            i000 = ix0 + i00;
            i001 = ix0 + i01;
            i010 = ix0 + i10;
            i011 = ix0 + i11;
            i100 = ix1 + i00;
            i101 = ix1 + i01;
            i110 = ix1 + i10;
            i111 = ix1 + i11;
        }
        reduce_t w000, w001, w010, w011, w100, w101, w110, w111;
        {
            reduce_t w00 = wy0 * wz0;
            reduce_t w01 = wy0 * wz1;
            reduce_t w10 = wy1 * wz0;
            reduce_t w11 = wy1 * wz1;
            w000 = wx0 * w00;
            w001 = wx0 * w01;
            w010 = wx0 * w10;
            w011 = wx0 * w11;
            w100 = wx1 * w00;
            w101 = wx1 * w01;
            w110 = wx1 * w10;
            w111 = wx1 * w11;
        }
        signed char f000, f001, f010, f011, f100, f101, f110, f111;
        {
            reduce_t f00 = fy0 * fz0;
            reduce_t f01 = fy0 * fz1;
            reduce_t f10 = fy1 * fz0;
            reduce_t f11 = fy1 * fz1;
            f000 = fx0 * f00;
            f001 = fx0 * f01;
            f010 = fx0 * f10;
            f011 = fx0 * f11;
            f100 = fx1 * f00;
            f101 = fx1 * f01;
            f110 = fx1 * f10;
            f111 = fx1 * f11;
        }

        bound::add(out, i000, w000, f000);
        bound::add(out, i001, w001, f001);
        bound::add(out, i010, w010, f010);
        bound::add(out, i011, w011, f011);
        bound::add(out, i100, w100, f100);
        bound::add(out, i101, w101, f101);
        bound::add(out, i110, w110, f110);
        bound::add(out, i111, w111, f111);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[3],
              const offset_t size[3],
              const offset_t stride[3],
              offset_t nc, offset_t osc, offset_t isc,
              offset_t osg)
    {
        offset_t    ix0, ix1, iy0, iy1, iz0, iz1;
        reduce_t    wx0, wx1, wy0, wy1, wz0, wz1;
        signed char fx0, fx1, fy0, fy1, fz0, fz1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        utils_z::index(loc[2], size[2], iz0, iz1, wz0, wz1, fz0, fz1);
        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        iz0 *= stride[2]; iz1 *= stride[2];
        offset_t i000, i001, i010, i011, i100, i101, i110, i111;
        {
            offset_t i00 = iy0 + iz0;
            offset_t i01 = iy0 + iz1;
            offset_t i10 = iy1 + iz0;
            offset_t i11 = iy1 + iz1;
            i000 = ix0 + i00;
            i001 = ix0 + i01;
            i010 = ix0 + i10;
            i011 = ix0 + i11;
            i100 = ix1 + i00;
            i101 = ix1 + i01;
            i110 = ix1 + i10;
            i111 = ix1 + i11;
        }
        reduce_t w000, w001, w010, w011, w100, w101, w110, w111;
        reduce_t wx00 = wy0 * wz0;
        reduce_t wx01 = wy0 * wz1;
        reduce_t wx10 = wy1 * wz0;
        reduce_t wx11 = wy1 * wz1;
        reduce_t wy00 = wx0 * wz0;
        reduce_t wy01 = wx0 * wz1;
        reduce_t wy10 = wx1 * wz0;
        reduce_t wy11 = wx1 * wz1;
        reduce_t wz00 = wx0 * wy0;
        reduce_t wz01 = wx0 * wy1;
        reduce_t wz10 = wx1 * wy0;
        reduce_t wz11 = wx1 * wy1;
        w000 = wx0 * wx00;
        w001 = wx0 * wx01;
        w010 = wx0 * wx10;
        w011 = wx0 * wx11;
        w100 = wx1 * wx00;
        w101 = wx1 * wx01;
        w110 = wx1 * wx10;
        w111 = wx1 * wx11;
        signed char f000, f001, f010, f011, f100, f101, f110, f111;
        {
            reduce_t f00 = fy0 * fz0;
            reduce_t f01 = fy0 * fz1;
            reduce_t f10 = fy1 * fz0;
            reduce_t f11 = fy1 * fz1;
            f000 = fx0 * f00;
            f001 = fx0 * f01;
            f010 = fx0 * f10;
            f011 = fx0 * f11;
            f100 = fx1 * f00;
            f101 = fx1 * f01;
            f110 = fx1 * f10;
            f111 = fx1 * f11;
        }

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t v000 = bound::cget<reduce_t>(inp, i000, f000);
            reduce_t v001 = bound::cget<reduce_t>(inp, i001, f001);
            reduce_t v010 = bound::cget<reduce_t>(inp, i010, f010);
            reduce_t v011 = bound::cget<reduce_t>(inp, i011, f011);
            reduce_t v100 = bound::cget<reduce_t>(inp, i100, f100);
            reduce_t v101 = bound::cget<reduce_t>(inp, i101, f101);
            reduce_t v110 = bound::cget<reduce_t>(inp, i110, f110);
            reduce_t v111 = bound::cget<reduce_t>(inp, i111, f111);
            out[0] = static_cast<scalar_t>(
                    - v000 * wx00 - v001 * wx01 - v010 * wx10 - v011 * wx11
                    + v100 * wx00 + v101 * wx01 + v110 * wx10 + v111 * wx11);
            out[osg] = static_cast<scalar_t>(
                    - v000 * wy00 - v001 * wy01 + v010 * wy00 + v011 * wy01
                    - v100 * wy10 - v101 * wy11 + v110 * wy10 + v111* wy11);
            out[osg * 2] = static_cast<scalar_t>(
                    - v000 * wz00 + v001 * wz00 - v010 * wz01 + v011 * wz01
                    - v100 * wz10 + v101 * wz10 - v110 * wz11 + v111 * wz11);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[3],
                       const offset_t size[3],
                       const offset_t stride_out[3],
                       const offset_t stride_inp[3],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        offset_t    ix0, ix1, iy0, iy1, iz0, iz1;
        reduce_t    wx0, wx1, wy0, wy1, wz0, wz1;
        signed char fx0, fx1, fy0, fy1, fz0, fz1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        utils_z::index(loc[2], size[2], iz0, iz1, wz0, wz1, fz0, fz1);
        offset_t i000, i001, i010, i011, i100, i101, i110, i111;
        offset_t o000, o001, o010, o011, o100, o101, o110, o111;
        {
            offset_t ox0 = ix0 * stride_out[0], ox1 = ix1 * stride_out[0];
            offset_t oy0 = iy0 * stride_out[1], oy1 = iy1 * stride_out[1];
            offset_t oz0 = iz0 * stride_out[2], oz1 = iz1 * stride_out[2];
            ix0 *= stride_inp[0]; ix1 *= stride_inp[0];
            iy0 *= stride_inp[1]; iy1 *= stride_inp[1];
            iz0 *= stride_inp[2]; iz1 *= stride_inp[2];
            offset_t i00 = iy0 + iz0;
            offset_t i01 = iy0 + iz1;
            offset_t i10 = iy1 + iz0;
            offset_t i11 = iy1 + iz1;
            offset_t o00 = oy0 + oz0;
            offset_t o01 = oy0 + oz1;
            offset_t o10 = oy1 + oz0;
            offset_t o11 = oy1 + oz1;
            i000 = ix0 + i00;
            i001 = ix0 + i01;
            i010 = ix0 + i10;
            i011 = ix0 + i11;
            i100 = ix1 + i00;
            i101 = ix1 + i01;
            i110 = ix1 + i10;
            i111 = ix1 + i11;
            o000 = ox0 + o00;
            o001 = ox0 + o01;
            o010 = ox0 + o10;
            o011 = ox0 + o11;
            o100 = ox1 + o00;
            o101 = ox1 + o01;
            o110 = ox1 + o10;
            o111 = ox1 + o11;
        }
        reduce_t w000, w001, w010, w011, w100, w101, w110, w111;
        reduce_t wx00 = wy0 * wz0;
        reduce_t wx01 = wy0 * wz1;
        reduce_t wx10 = wy1 * wz0;
        reduce_t wx11 = wy1 * wz1;
        reduce_t wy00 = wx0 * wz0;
        reduce_t wy01 = wx0 * wz1;
        reduce_t wy10 = wx1 * wz0;
        reduce_t wy11 = wx1 * wz1;
        reduce_t wz00 = wx0 * wy0;
        reduce_t wz01 = wx0 * wy1;
        reduce_t wz10 = wx1 * wy0;
        reduce_t wz11 = wx1 * wy1;
        w000 = wx0 * wx00;
        w001 = wx0 * wx01;
        w010 = wx0 * wx10;
        w011 = wx0 * wx11;
        w100 = wx1 * wx00;
        w101 = wx1 * wx01;
        w110 = wx1 * wx10;
        w111 = wx1 * wx11;
        signed char f000, f001, f010, f011, f100, f101, f110, f111;
        {
            reduce_t f00 = fy0 * fz0;
            reduce_t f01 = fy0 * fz1;
            reduce_t f10 = fy1 * fz0;
            reduce_t f11 = fy1 * fz1;
            f000 = fx0 * f00;
            f001 = fx0 * f01;
            f010 = fx0 * f10;
            f011 = fx0 * f11;
            f100 = fx1 * f00;
            f101 = fx1 * f01;
            f110 = fx1 * f10;
            f111 = fx1 * f11;
        }

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        reduce_t accz = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, o000, gval * w000, f000);
            bound::add(out, o001, gval * w001, f001);
            bound::add(out, o010, gval * w010, f010);
            bound::add(out, o011, gval * w011, f011);
            bound::add(out, o100, gval * w100, f100);
            bound::add(out, o101, gval * w101, f101);
            bound::add(out, o110, gval * w110, f110);
            bound::add(out, o111, gval * w111, f111);
            // compute input spatial gradient
            reduce_t v000 = bound::cget<reduce_t>(inp, i000, f000);
            reduce_t v001 = bound::cget<reduce_t>(inp, i001, f001);
            reduce_t v010 = bound::cget<reduce_t>(inp, i010, f010);
            reduce_t v011 = bound::cget<reduce_t>(inp, i011, f011);
            reduce_t v100 = bound::cget<reduce_t>(inp, i100, f100);
            reduce_t v101 = bound::cget<reduce_t>(inp, i101, f101);
            reduce_t v110 = bound::cget<reduce_t>(inp, i110, f110);
            reduce_t v111 = bound::cget<reduce_t>(inp, i111, f111);
            accx += gval * (
                    - v000 * wx00 - v001 * wx01 - v010 * wx10 - v011 * wx11
                    + v100 * wx00 + v101 * wx01 + v110 * wx10 + v111 * wx11);
            accy += gval * (
                    - v000 * wy00 - v001 * wy01 + v010 * wy00 + v011 * wy01
                    - v100 * wy10 - v101 * wy11 + v110 * wy10 + v111 * wy11);
            accz += gval * (
                    - v000 * wz00 + v001 * wz00 - v010 * wz01 + v011 * wz01
                    - v100 * wz10 + v101 * wz10 - v110 * wz11 + v111 * wz11);
        }
        gout[0]       = static_cast<scalar_t>(accx);
        gout[osg]     = static_cast<scalar_t>(accy);
        gout[osg * 2] = static_cast<scalar_t>(accz);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[3],
                       const offset_t size[3],
                       const offset_t stride[3],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        offset_t    ix0, ix1, iy0, iy1, iz0, iz1;
        reduce_t    wx0, wx1, wy0, wy1, wz0, wz1;
        signed char fx0, fx1, fy0, fy1, fz0, fz1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        utils_z::index(loc[2], size[2], iz0, iz1, wz0, wz1, fz0, fz1);
        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        iz0 *= stride[2]; iz1 *= stride[2];
        offset_t i000, i001, i010, i011, i100, i101, i110, i111;
        {
            offset_t i00 = iy0 + iz0;
            offset_t i01 = iy0 + iz1;
            offset_t i10 = iy1 + iz0;
            offset_t i11 = iy1 + iz1;
            i000 = ix0 + i00;
            i001 = ix0 + i01;
            i010 = ix0 + i10;
            i011 = ix0 + i11;
            i100 = ix1 + i00;
            i101 = ix1 + i01;
            i110 = ix1 + i10;
            i111 = ix1 + i11;
        }
        reduce_t w000, w001, w010, w011, w100, w101, w110, w111;
        reduce_t wx00 = wy0 * wz0;
        reduce_t wx01 = wy0 * wz1;
        reduce_t wx10 = wy1 * wz0;
        reduce_t wx11 = wy1 * wz1;
        reduce_t wy00 = wx0 * wz0;
        reduce_t wy01 = wx0 * wz1;
        reduce_t wy10 = wx1 * wz0;
        reduce_t wy11 = wx1 * wz1;
        reduce_t wz00 = wx0 * wy0;
        reduce_t wz01 = wx0 * wy1;
        reduce_t wz10 = wx1 * wy0;
        reduce_t wz11 = wx1 * wy1;
        w000 = wx0 * wx00;
        w001 = wx0 * wx01;
        w010 = wx0 * wx10;
        w011 = wx0 * wx11;
        w100 = wx1 * wx00;
        w101 = wx1 * wx01;
        w110 = wx1 * wx10;
        w111 = wx1 * wx11;
        signed char f000, f001, f010, f011, f100, f101, f110, f111;
        {
            reduce_t f00 = fy0 * fz0;
            reduce_t f01 = fy0 * fz1;
            reduce_t f10 = fy1 * fz0;
            reduce_t f11 = fy1 * fz1;
            f000 = fx0 * f00;
            f001 = fx0 * f01;
            f010 = fx0 * f10;
            f011 = fx0 * f11;
            f100 = fx1 * f00;
            f101 = fx1 * f01;
            f110 = fx1 * f10;
            f111 = fx1 * f11;
        }

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        reduce_t accz = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            // pull incoming gradient
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(ginp, i000, f000) * w000
                    + bound::cget<reduce_t>(ginp, i001, f001) * w001
                    + bound::cget<reduce_t>(ginp, i010, f010) * w010
                    + bound::cget<reduce_t>(ginp, i011, f011) * w011
                    + bound::cget<reduce_t>(ginp, i100, f100) * w100
                    + bound::cget<reduce_t>(ginp, i101, f101) * w101
                    + bound::cget<reduce_t>(ginp, i110, f110) * w110
                    + bound::cget<reduce_t>(ginp, i111, f111) * w111);
            // compute input spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            reduce_t v000 = bound::cget<reduce_t>(ginp, i000, f000);
            reduce_t v001 = bound::cget<reduce_t>(ginp, i001, f001);
            reduce_t v010 = bound::cget<reduce_t>(ginp, i010, f010);
            reduce_t v011 = bound::cget<reduce_t>(ginp, i011, f011);
            reduce_t v100 = bound::cget<reduce_t>(ginp, i100, f100);
            reduce_t v101 = bound::cget<reduce_t>(ginp, i101, f101);
            reduce_t v110 = bound::cget<reduce_t>(ginp, i110, f110);
            reduce_t v111 = bound::cget<reduce_t>(ginp, i111, f111);
            accx += val * (
                    - v000 * wx00 - v001 * wx01 - v010 * wx10 - v011 * wx11
                    + v100 * wx00 + v101 * wx01 + v110 * wx10 + v111 * wx11);
            accy += val * (
                    - v000 * wy00 - v001 * wy01 + v010 * wy00 + v011 * wy01
                    - v100 * wy10 - v101 * wy11 + v110 * wy10 + v111 * wy11);
            accz += val * (
                    - v000 * wz00 + v001 * wz00 - v010 * wz01 + v011 * wz01
                    - v100 * wz10 + v101 * wz10 - v110 * wz11 + v111 * wz11);
        }
        gout[0]       = static_cast<scalar_t>(accx);
        gout[osg]     = static_cast<scalar_t>(accy);
        gout[osg * 2] = static_cast<scalar_t>(accz);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count_backward(scalar_t * gout, const scalar_t * ginp,
                       const reduce_t loc[3],
                       const offset_t size[3],
                       const offset_t stride[3],
                        offset_t osg)
    {
        offset_t    ix0, ix1, iy0, iy1, iz0, iz1;
        reduce_t    wx0, wx1, wy0, wy1, wz0, wz1;
        signed char fx0, fx1, fy0, fy1, fz0, fz1;
        utils_x::index(loc[0], size[0], ix0, ix1, wx0, wx1, fx0, fx1);
        utils_y::index(loc[1], size[1], iy0, iy1, wy0, wy1, fy0, fy1);
        utils_z::index(loc[2], size[2], iz0, iz1, wz0, wz1, fz0, fz1);
        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        iz0 *= stride[2]; iz1 *= stride[2];
        offset_t i000, i001, i010, i011, i100, i101, i110, i111;
        {
            offset_t i00 = iy0 + iz0;
            offset_t i01 = iy0 + iz1;
            offset_t i10 = iy1 + iz0;
            offset_t i11 = iy1 + iz1;
            i000 = ix0 + i00;
            i001 = ix0 + i01;
            i010 = ix0 + i10;
            i011 = ix0 + i11;
            i100 = ix1 + i00;
            i101 = ix1 + i01;
            i110 = ix1 + i10;
            i111 = ix1 + i11;
        }
        reduce_t w000, w001, w010, w011, w100, w101, w110, w111;
        reduce_t wx00 = wy0 * wz0;
        reduce_t wx01 = wy0 * wz1;
        reduce_t wx10 = wy1 * wz0;
        reduce_t wx11 = wy1 * wz1;
        reduce_t wy00 = wx0 * wz0;
        reduce_t wy01 = wx0 * wz1;
        reduce_t wy10 = wx1 * wz0;
        reduce_t wy11 = wx1 * wz1;
        reduce_t wz00 = wx0 * wy0;
        reduce_t wz01 = wx0 * wy1;
        reduce_t wz10 = wx1 * wy0;
        reduce_t wz11 = wx1 * wy1;
        w000 = wx0 * wx00;
        w001 = wx0 * wx01;
        w010 = wx0 * wx10;
        w011 = wx0 * wx11;
        w100 = wx1 * wx00;
        w101 = wx1 * wx01;
        w110 = wx1 * wx10;
        w111 = wx1 * wx11;
        signed char f000, f001, f010, f011, f100, f101, f110, f111;
        {
            reduce_t f00 = fy0 * fz0;
            reduce_t f01 = fy0 * fz1;
            reduce_t f10 = fy1 * fz0;
            reduce_t f11 = fy1 * fz1;
            f000 = fx0 * f00;
            f001 = fx0 * f01;
            f010 = fx0 * f10;
            f011 = fx0 * f11;
            f100 = fx1 * f00;
            f101 = fx1 * f01;
            f110 = fx1 * f10;
            f111 = fx1 * f11;
        }

        // compute input spatial gradient
        reduce_t v000 = bound::cget<reduce_t>(ginp, i000, f000);
        reduce_t v001 = bound::cget<reduce_t>(ginp, i001, f001);
        reduce_t v010 = bound::cget<reduce_t>(ginp, i010, f010);
        reduce_t v011 = bound::cget<reduce_t>(ginp, i011, f011);
        reduce_t v100 = bound::cget<reduce_t>(ginp, i100, f100);
        reduce_t v101 = bound::cget<reduce_t>(ginp, i101, f101);
        reduce_t v110 = bound::cget<reduce_t>(ginp, i110, f110);
        reduce_t v111 = bound::cget<reduce_t>(ginp, i111, f111);
        gout[0] = static_cast<scalar_t>(
                - v000 * wx00 - v001 * wx01 - v010 * wx10 - v011 * wx11
                + v100 * wx00 + v101 * wx01 + v110 * wx10 + v111 * wx11);
        gout[osg] = static_cast<scalar_t>(
                - v000 * wy00 - v001 * wy01 + v010 * wy00 + v011 * wy01
                - v100 * wy10 - v101 * wy11 + v110 * wy10 + v111 * wy11);
        gout[osg * 2] = static_cast<scalar_t>(
                - v000 * wz00 + v001 * wz00 - v010 * wz01 + v011 * wz01
                - v100 * wz10 + v101 * wz10 - v110 * wz11 + v111 * wz11);
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       reduce_t y, offset_t ny, offset_t osy, offset_t isy,
                       reduce_t z, offset_t nz, offset_t osz, offset_t isz,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc,
                       offset_t osg, offset_t isg)
    {
        gout[0]       = static_cast<scalar_t>(0);
        gout[osg]     = static_cast<scalar_t>(0);
        gout[osg * 2] = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }
};


/***********************************************************************
 *
 *                                 ANY
 *
 **********************************************************************/
template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ>
struct PushPull<three, IX, BX, IY, BY, IZ, BZ> {
    using utils_x = PushPullAnyUtils<IX, BX>;
    using utils_y = PushPullAnyUtils<IY, BY>;
    using utils_z = PushPullAnyUtils<IZ, BZ>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[3],
              const offset_t size[3],
              const offset_t stride[3],
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::index(loc[0], size[0], ix, wx, fx);
        offset_t ly = utils_y::index(loc[1], size[1], iy, wy, fy);
        offset_t lz = utils_z::index(loc[2], size[2], iz, wz, fz);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;
        for (offset_t i = 0, s = stride[2]; i <= lz; ++i)
            iz[i] *= s;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= lx; ++i)
            for (offset_t j = 0; j <= ly; ++j)
            for (offset_t k = 0; k <= lz; ++k)
                acc += bound::cget<reduce_t>(
                    inp, ix[i] + iy[j] + iz[k], fx[i] * fy[j] * fz[k])
                    * (wx[i] * wy[j] * wz[k]);
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[3],
              const offset_t size[3],
              const offset_t stride[3],
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::index(loc[0], size[0], ix, wx, fx);
        offset_t ly = utils_y::index(loc[1], size[1], iy, wy, fy);
        offset_t lz = utils_z::index(loc[2], size[2], iz, wz, fz);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;
        for (offset_t i = 0, s = stride[2]; i <= lz; ++i)
            iz[i] *= s;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            for (offset_t i = 0; i <= lx; ++i)
            for (offset_t j = 0; j <= ly; ++j)
            for (offset_t k = 0; k <= lz; ++k)
                bound::add(out, ix[i] + iy[j] + iz[k],
                           val * (wx[i] * wy[j] * wz[k]),
                           fx[i] * fy[j] * fz[k]);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count(scalar_t * out,
               const reduce_t loc[3],
               const offset_t size[3],
               const offset_t stride[3])
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::index(loc[0], size[0], ix, wx, fx);
        offset_t ly = utils_y::index(loc[1], size[1], iy, wy, fy);
        offset_t lz = utils_z::index(loc[2], size[2], iz, wz, fz);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;
        for (offset_t i = 0, s = stride[2]; i <= lz; ++i)
            iz[i] *= s;

        for (offset_t i = 0; i <= lx; ++i)
        for (offset_t j = 0; j <= ly; ++j)
        for (offset_t k = 0; k <= lz; ++k)
            bound::add(out, ix[i] + iy[j] + iz[k],
                       wx[i] * wy[j] * wz[k],
                       fx[i] * fy[j] * fz[k]);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad(scalar_t * out, const scalar_t * inp,
              const reduce_t loc[3],
              const offset_t size[3],
              const offset_t stride[3],
              offset_t nc, offset_t osc, offset_t isc, offset_t osg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        reduce_t    gx[8], gy[8], gz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx);
        offset_t ly = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy);
        offset_t lz = utils_z::gindex(loc[2], size[2], iz, wz, gz, fz);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;
        for (offset_t i = 0, s = stride[2]; i <= lz; ++i)
            iz[i] *= s;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t accx = static_cast<reduce_t>(0);
            reduce_t accy = static_cast<reduce_t>(0);
            reduce_t accz = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= lx; ++i)
            for (offset_t j = 0; j <= ly; ++j)
            for (offset_t k = 0; k <= lz; ++k) {
                reduce_t val = bound::cget<reduce_t>(
                    inp, ix[i] + iy[j] + iz[k], fx[i] * fy[j] * fz[k]);
                accx += val * (gx[i] * wy[j] * wz[k]);
                accy += val * (wx[i] * gy[j] * wz[k]);
                accz += val * (wx[i] * wy[j] * gz[k]);
            }
            out[0]       = static_cast<scalar_t>(accx);
            out[osg]     = static_cast<scalar_t>(accy);
            out[osg * 2] = static_cast<scalar_t>(accz);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void pull_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[3],
                       const offset_t size[3],
                       const offset_t stride_out[3],
                       const offset_t stride_inp[3],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        reduce_t    gx[8], gy[8], gz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx);
        offset_t ly = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy);
        offset_t lz = utils_z::gindex(loc[2], size[2], iz, wz, gz, fz);
        offset_t osx = stride_out[0], osy = stride_out[1], osz = stride_out[2];
        offset_t isx = stride_inp[0], isy = stride_inp[1], isz = stride_inp[2];

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        reduce_t accz = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            reduce_t gval  = static_cast<reduce_t>(*ginp);
            reduce_t accx1 = static_cast<reduce_t>(0);
            reduce_t accy1 = static_cast<reduce_t>(0);
            reduce_t accz1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= lx; ++i) {
                offset_t ixo = ix[i] * osx;
                offset_t ixi = ix[i] * isx;
                signed char ffx = fx[i];
                reduce_t wwx = wx[i];
                reduce_t ggx = gx[i];
                for (offset_t j = 0; j <= ly; ++j) {
                    offset_t iyo = ixo + iy[j] * osy;
                    offset_t iyi = ixi + iy[j] * isy;
                    signed char ffy = ffx * fy[j];
                    reduce_t wwy = wy[j];
                    reduce_t ggy = gy[j];
                    for (offset_t k = 0; k <= lz; ++k) {
                        offset_t izo = iyo + iz[k] * osz;
                        offset_t izi = iyi + iz[k] * isz;
                        signed char ff = ffy * fz[k];
                        reduce_t wwz = wz[k];
                        reduce_t ggz = gz[k];
                        // push incoming gradient
                        bound::add(out, izo, gval * (wwx * wwy * wwz), ff);
                        // compute input spatial gradient
                        reduce_t val = bound::cget<reduce_t>(inp, izi, ff);
                        accx1 += val * (ggx * wwy * wwz);
                        accy1 += val * (wwx * ggy * wwz);
                        accz1 += val * (wwx * wwy * ggz);
                    }
                }
            }
            accx += gval * accx1;
            accy += gval * accy1;
            accz += gval * accz1;
        }
        gout[0]       = static_cast<scalar_t>(accx);
        gout[osg]     = static_cast<scalar_t>(accy);
        gout[osg * 2] = static_cast<scalar_t>(accz);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void push_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[3],
                       const offset_t size[3],
                       const offset_t stride[3],
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        reduce_t    gx[8], gy[8], gz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx);
        offset_t ly = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy);
        offset_t lz = utils_z::gindex(loc[2], size[2], iz, wz, gz, fz);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;
        for (offset_t i = 0, s = stride[2]; i <= lz; ++i)
            iz[i] *= s;

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        reduce_t accz = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            reduce_t val   = static_cast<reduce_t>(*inp);
            reduce_t acc1  = static_cast<reduce_t>(0);
            reduce_t accx2 = static_cast<reduce_t>(0);
            reduce_t accy2 = static_cast<reduce_t>(0);
            reduce_t accz2 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= lx; ++i)
            for (offset_t j = 0; j <= ly; ++j)
            for (offset_t k = 0; k <= lz; ++k) {
                reduce_t gval = bound::cget<reduce_t>(
                    ginp, ix[i] + iy[j] + iz[k], fx[i] * fy[j] * fz[k]);
                // pull incoming gradient
                acc1 += gval * (wx[i] * wy[j] * wz[k]);
                // compute incoming gradient spatial gradient
                accx2 += gval * (gx[i] * wy[j] * wz[k]);
                accy2 += gval * (wx[i] * gy[j] * wz[k]);
                accz2 += gval * (wx[i] * wy[j] * gz[k]);
            }
            *out = static_cast<scalar_t>(acc1);
            accx += val * accx2;
            accy += val * accy2;
            accz += val * accz2;
        }
        gout[0]       = static_cast<scalar_t>(accx);
        gout[osg]     = static_cast<scalar_t>(accy);
        gout[osg * 2] = static_cast<scalar_t>(accz);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void count_backward(scalar_t * gout, const scalar_t * ginp,
                        const reduce_t loc[3],
                        const offset_t size[3],
                        const offset_t stride[3],
                        offset_t osg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        reduce_t    gx[8], gy[8], gz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx);
        offset_t ly = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy);
        offset_t lz = utils_z::gindex(loc[2], size[2], iz, wz, gz, fz);
        for (offset_t i = 0, s = stride[0]; i <= lx; ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i <= ly; ++i)
            iy[i] *= s;
        for (offset_t i = 0, s = stride[2]; i <= lz; ++i)
            iz[i] *= s;

        // compute input spatial gradient
        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        reduce_t accz = static_cast<reduce_t>(0);
        for (offset_t i = 0; i <= lx; ++i)
        for (offset_t j = 0; j <= ly; ++j)
        for (offset_t k = 0; k <= lz; ++k) {
            reduce_t val = bound::cget<reduce_t>(
                ginp, ix[i] + iy[j] + iz[k], fx[i] * fy[j] * fz[k]);
            accx += val * (gx[i] * wy[j] * wz[k]);
            accy += val * (wx[i] * gy[j] * wz[k]);
            accz += val * (wx[i] * wy[j] * gz[k]);
        }
        gout[0]       = static_cast<scalar_t>(accx);
        gout[osg]     = static_cast<scalar_t>(accy);
        gout[osg * 2] = static_cast<scalar_t>(accz);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    __device__ static inline
    void grad_backward(scalar_t * out, scalar_t * gout,
                       const scalar_t * inp, const scalar_t * ginp,
                       const reduce_t loc[3],
                       const offset_t size[3],
                       const offset_t stride_out[3],
                       const offset_t stride_inp[3],
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc,
                       offset_t osg, offset_t isg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        reduce_t    gx[8], gy[8], gz[8];
        reduce_t    hx[8], hy[8], hz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::hindex(loc[0], size[0], ix, wx, gx, hx, fx);
        offset_t ly = utils_y::hindex(loc[1], size[1], iy, wy, gy, hy, fy);
        offset_t lz = utils_z::hindex(loc[2], size[2], iz, wz, gz, hz, fz);
        offset_t osx = stride_out[0], osy = stride_out[1], osz = stride_out[2];
        offset_t isx = stride_inp[0], isy = stride_inp[1], isz = stride_inp[2];

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        reduce_t accz = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t gvalx = static_cast<reduce_t>(ginp[0]);
            reduce_t gvaly = static_cast<reduce_t>(ginp[isg]);
            reduce_t gvalz = static_cast<reduce_t>(ginp[isg * 2]);
            reduce_t accxx1 = static_cast<reduce_t>(0);
            reduce_t accyy1 = static_cast<reduce_t>(0);
            reduce_t acczz1 = static_cast<reduce_t>(0);
            reduce_t accxy1 = static_cast<reduce_t>(0);
            reduce_t accxz1 = static_cast<reduce_t>(0);
            reduce_t accyz1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i <= lx; ++i)
            for (offset_t j = 0; j <= ly; ++j)
            for (offset_t k = 0; k <= lz; ++k) {
                signed char f = fx[i] * fy[j] * fz[k];
                // push incoming gradient
                reduce_t oval = gvalx * (gx[i] * wy[j] * wz[k])
                              + gvaly * (wx[i] * gy[j] * wz[k])
                              + gvalz * (wx[i] * wy[j] * gz[k]);
                bound::add(out, ix[i] * osx + iy[j] * osy + iz[k] * osz,
                           oval, f);
                // compute input spatial hessian
                reduce_t ival = bound::cget<reduce_t>(
                    inp, ix[i] * isx + iy[j] * isy + iz[k] * isz, f);
                accxx1 += ival * hx[i] * wy[j] * wz[k];
                accyy1 += ival * wx[i] * hy[j] * wz[k];
                acczz1 += ival * wx[i] * wy[j] * hz[k];
                accxy1 += ival * gx[i] * gy[j] * wz[k];
                accxz1 += ival * gx[i] * wy[j] * gz[k];
                accyz1 += ival * wx[i] * gy[j] * gz[k];
            }
            accx += gvalx * accxx1 + gvaly * accxy1 + gvalz * accxz1;
            accy += gvaly * accyy1 + gvalx * accxy1 + gvalz * accyz1;
            accz += gvalz * acczz1 + gvalx * accxz1 + gvaly * accyz1;
        }
        gout[0]       = static_cast<scalar_t>(accx);
        gout[osg]     = static_cast<scalar_t>(accy);
        gout[osg * 2] = static_cast<scalar_t>(accz);
    }
};

} // namespace pushpull
} // namespace jf

#endif // JF_PUSHPULL_3D
