/***********************************************************************
 *
 *                                  ND
 *
 **********************************************************************/
#ifndef JF_PUSHPULL_ND
#define JF_PUSHPULL_ND
#include "../cuda_switch.h"
#include "../spline.h"
#include "../bounds.h"
#include "utils.h"

// TODO

namespace jf {
namespace pushpull {


/***********************************************************************
 *
 *                                 ANY
 *
 **********************************************************************/
template <int D, bool ABS>
struct PushPull<D,Z,B0,Z,B0,Z,B0,ABS> {
    using maybe = PushPullMaybe<ABS>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull(scalar_t * out, scalar_t * inp,
              const offset_t * coord, const offset_t * size, const offset_t * stride,
              const spline::type * inter, const bound::type * bnd,
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    i[8*D];
        reduce_t    w[8*D];
        signed char f[8*D];
        offset_t    l[8*D];
        for (int d=0; d<D; ++d) {
            reduce_t    *wd = w + 8*d;
            offset_t    *id = i + 8*d;
            signed char *sd = s + 8*d;
            reduce_t x = coord[d];
            offset_t b0, b1;
            spline::bounds(inter[d], x, b0, b1);
            l[d] = b1-b0;
            for (offset_t b = b0; b <= b1; ++b) {
                *(wd++) = spline::fastweight(inter[d], fabs(x - b));
                *(sd++) = bound::sign(bnd[d], b, size[d]);
                *(id++) = bound::index(bnd[d], b, size[d]);
            }
        }

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            offset_t    offsets[D];
            signed char signs[D];
            scalar_t    weights[D];
            reduce_t acc = static_cast<reduce_t>(0);
            for (int d=0; d<D; ++d) {
                reduce_t    *wd = w + 8*d;
                offset_t    *id = i + 8*d;
                signed char *sd = s + 8*d;
                for (offset_t k = 0; k <= db[d]; ++k) {
                    offsets[d] = (d > 0 ? offsets[d-1] : static_cast<offset_t>(0))
                               + id[k] * stride[d];
                    signs[d]   = (d > 0 ? signs[d-1]   : static_cast<signed char>(1))
                               * sd[k];
                    weights[d] = (d > 0 ? weights[d-1] : static_cast<reduce_t>(1))
                               * wd[k];
                    if (d == D-1)
                        acc += bound::cget<reduce_t>(inp, offsets[D-1], signs[D-1]) * weights[D-1];
                }
            }
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void push(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              reduce_t y, offset_t ny, offset_t sy,
              reduce_t z, offset_t nz, offset_t sz,
              offset_t nc, offset_t osc, offset_t isc)
    {
        // Precompute weights and indices
        offset_t    i[8*D];
        reduce_t    w[8*D];
        signed char f[8*D];
        offset_t    l[8*D];
        for (int d=0; d<D; ++d) {
            reduce_t    *wd = w + 8*d;
            offset_t    *id = i + 8*d;
            signed char *sd = s + 8*d;
            reduce_t x = coord[d];
            offset_t b0, b1;
            spline::bounds(inter[d], x, b0, b1);
            l[d] = b1-b0;
            for (offset_t b = b0; b <= b1; ++b) {
                *(wd++) = spline::fastweight(inter[d], fabs(x - b));
                *(sd++) = bound::sign(bnd[d], b, size[d]);
                *(id++) = bound::index(bnd[d], b, size[d]);
            }
        }

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t val = static_cast<reduce_t>(*inp);
            offset_t    offsets[D];
            signed char signs[D];
            scalar_t    weights[D];
            for (int d=0; d<D; ++d) {
                reduce_t    *wd = w + 8*d;
                offset_t    *id = i + 8*d;
                signed char *sd = s + 8*d;
                for (offset_t k = 0; k <= db[d]; ++k) {
                    offsets[d] = (d > 0 ? offsets[d-1] : static_cast<offset_t>(0))
                               + id[k] * stride[d];
                    signs[d]   = (d > 0 ? signs[d-1]   : static_cast<signed char>(1))
                               * sd[k];
                    weights[d] = (d > 0 ? weights[d-1] : static_cast<reduce_t>(1))
                               * wd[k];
                    if (d == D-1)
                        bound::add(out, offsets[D-1], val * weights[D-1], signs[D-1]);
                }
            }
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void count(scalar_t * out,
               reduce_t x, offset_t nx, offset_t sx,
               reduce_t y, offset_t ny, offset_t sy,
               reduce_t z, offset_t nz, offset_t sz)
    {
        // Precompute weights and indices
        offset_t    i[8*D];
        reduce_t    w[8*D];
        signed char f[8*D];
        offset_t    l[8*D];
        for (int d=0; d<D; ++d) {
            reduce_t    *wd = w + 8*d;
            offset_t    *id = i + 8*d;
            signed char *sd = s + 8*d;
            reduce_t x = coord[d];
            offset_t b0, b1;
            spline::bounds(inter[d], x, b0, b1);
            l[d] = b1-b0;
            for (offset_t b = b0; b <= b1; ++b) {
                *(wd++) = spline::fastweight(inter[d], fabs(x - b));
                *(sd++) = bound::sign(bnd[d], b, size[d]);
                *(id++) = bound::index(bnd[d], b, size[d]);
            }
        }

        offset_t    offsets[D];
        signed char signs[D];
        scalar_t    weights[D];
        for (int d=0; d<D; ++d) {
            reduce_t    *wd = w + 8*d;
            offset_t    *id = i + 8*d;
            signed char *sd = s + 8*d;
            for (offset_t k = 0; k <= db[d]; ++k) {
                offsets[d] = (d > 0 ? offsets[d-1] : static_cast<offset_t>(0))
                           + id[k] * stride[d];
                signs[d]   = (d > 0 ? signs[d-1]   : static_cast<signed char>(1))
                           * sd[k];
                weights[d] = (d > 0 ? weights[d-1] : static_cast<reduce_t>(1))
                           * wd[k];
                if (d == D-1)
                    bound::add(out, offsets[D-1], weights[D-1], signs[D-1]);
            }
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void grad(scalar_t * out, scalar_t * inp,
              reduce_t x, offset_t nx, offset_t sx,
              reduce_t y, offset_t ny, offset_t sy,
              reduce_t z, offset_t nz, offset_t sz,
              offset_t nc, offset_t osc, offset_t isc, offset_t osg)
    {
        // Precompute weights and indices
        offset_t    i[8*D];
        reduce_t    w[8*D];
        reduce_t    g[8*D];
        signed char f[8*D];
        offset_t    l[8*D];
        for (int d=0; d<D; ++d) {
            reduce_t    *wd = w + 8*d;
            reduce_t    *gd = g + 8*d;
            offset_t    *id = i + 8*d;
            signed char *sd = s + 8*d;
            reduce_t x = coord[d];
            offset_t b0, b1;
            spline::bounds(inter[d], x, b0, b1);
            l[d] = b1-b0;
            for (offset_t b = b0; b <= b1; ++b) {
                reduce_t dist = fabs(x - b);
                *(wd++) = spline::fastweight(inter[d], dist);
                *(gd++) = maybe::fabs(spline::fastgrad(inter[d], dist));
                *(sd++) = bound::sign(bnd[d], b, size[d]);
                *(id++) = bound::index(bnd[d], b, size[d]);
            }
        }

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            offset_t    offsets[D];
            signed char signs[D];
            scalar_t    weights[D];
            scalar_t    grads[D];
            reduce_t    acc[D];
            for (int d=0; d<D; ++d)
                acc[d] = static_cast<reduce_t>(0);
            for (int d=0; d<D; ++d) {
                reduce_t    *wd = w + 8*d;
                reduce_t    *gd = g + 8*d;
                offset_t    *id = i + 8*d;
                signed char *sd = s + 8*d;
                for (offset_t k = 0; k <= db[d]; ++k) {
                    offsets[d] = (d > 0 ? offsets[d-1] : static_cast<offset_t>(0))
                               + id[k] * stride[d];
                    signs[d]   = (d > 0 ? signs[d-1]   : static_cast<signed char>(1))
                               * sd[k];
                    weights[d] = (d > 0 ? weights[d-1] : static_cast<reduce_t>(1))
                               * wd[k];
                    grads[d] = (d > 0 ? grads[d-1] : static_cast<reduce_t>(1))
                               * gd[k];
                    if (d == D-1) {
                        reduce_t val = bound::cget<reduce_t>(inp, offsets[D-1], signs[D-1]);
                        for (int dd = 0; dd < D; ++dd) {
                            acc[dd] += val * (g[8*d + ] * weights[D-1];
                        }
                    }
                }
            }
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    static __device__
    void pull_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       reduce_t y, offset_t ny, offset_t osy, offset_t isy,
                       reduce_t z, offset_t nz, offset_t osz, offset_t isz,
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        reduce_t    gx[8], gy[8], gz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::gindex(x, nx, ix, wx, gx, fx);
        offset_t ly = utils_y::gindex(y, ny, iy, wy, gy, fy);
        offset_t lz = utils_z::gindex(z, nz, iz, wz, gz, fz);

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
                offset_t ffx = fx[i];
                offset_t wwx = wx[i];
                offset_t ggx = gx[i];
                for (offset_t j = 0; j <= ly; ++j) {
                    offset_t iyo = ixo + iy[j] * osy;
                    offset_t iyi = ixi + iy[j] * isy;
                    offset_t ffy = ffx * fy[j];
                    offset_t wwy = wy[j];
                    offset_t ggy = gy[j];
                    for (offset_t k = 0; k <= lz; ++k) {
                        offset_t izo = iyo + iz[k] * osz;
                        offset_t izi = iyi + iz[k] * isz;
                        offset_t  ff = ffy * fz[k];
                        offset_t wwz = wz[k];
                        offset_t ggz = gz[k];
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
    static __device__
    void push_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t sx,
                       reduce_t y, offset_t ny, offset_t sy,
                       reduce_t z, offset_t nz, offset_t sz,
                       offset_t nc, offset_t osc, offset_t isc,
                       offset_t osg, offset_t isg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        reduce_t    gx[8], gy[8], gz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::gindex(x, nx, ix, wx, gx, fx);
        offset_t ly = utils_y::gindex(y, ny, iy, wy, gy, fy);
        offset_t lz = utils_z::gindex(z, nz, iz, wz, gz, fz);
        for (offset_t i = 0; i <= lx; ++i)
            ix[i] *= sx;
        for (offset_t i = 0; i <= ly; ++i)
            iy[i] *= sy;
        for (offset_t i = 0; i <= lz; ++i)
            iz[i] *= sz;

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
    static __device__
    void count_backward(scalar_t * gout, scalar_t * ginp,
                        reduce_t x, offset_t nx, offset_t sx,
                        reduce_t y, offset_t ny, offset_t sy,
                        reduce_t z, offset_t nz, offset_t sz,
                        offset_t osg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        reduce_t    gx[8], gy[8], gz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::gindex(x, nx, ix, wx, gx, fx);
        offset_t ly = utils_y::gindex(y, ny, iy, wy, gy, fy);
        offset_t lz = utils_z::gindex(z, nz, iz, wz, gz, fz);
        for (offset_t i = 0; i <= lx; ++i)
            ix[i] *= sx;
        for (offset_t i = 0; i <= ly; ++i)
            iy[i] *= sy;
        for (offset_t i = 0; i <= lz; ++i)
            iz[i] *= sz;

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
    static __device__
    void grad_backward(scalar_t * out, scalar_t * gout,
                       scalar_t * inp, scalar_t * ginp,
                       reduce_t x, offset_t nx, offset_t osx, offset_t isx,
                       reduce_t y, offset_t ny, offset_t osy, offset_t isy,
                       reduce_t z, offset_t nz, offset_t osz, offset_t isz,
                       offset_t nc, offset_t osc, offset_t isc, offset_t gsc,
                       offset_t osg, offset_t isg)
    {
        // Precompute weights and indices
        offset_t    ix[8], iy[8], iz[8];
        reduce_t    wx[8], wy[8], wz[8];
        reduce_t    gx[8], gy[8], gz[8];
        reduce_t    hx[8], hy[8], hz[8];
        signed char fx[8], fy[8], fz[8];
        offset_t lx = utils_x::hindex(x, nx, ix, wx, gx, hx, fx);
        offset_t ly = utils_y::hindex(y, ny, iy, wy, gy, hy, fy);
        offset_t lz = utils_z::hindex(z, nz, iz, wz, gz, hz, fz);

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

#endif JF_PUSHPULL_ND
