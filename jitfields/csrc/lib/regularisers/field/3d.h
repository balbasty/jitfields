#ifndef JF_REGULARISERS_1D
#define JF_REGULARISERS_1D
#include "../../cuda_swap.h"
#include "../../bounds.h"
#include "../../utils.h"
#include "utils.h"

namespace jf {
namespace reg_field {

template <bound::type BX, bound::type BY, bound::type BZ>
class RegFieldBase<three, BX, BY, BZ> {
    bound_utils_x = bound::utils<BX>;
    bound_utils_y = bound::utils<BY>;
    bound_utils_z = bound::utils<BZ>;

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_absolute(scalar_t * out, const scalar_t * inp,
                          offset_t nc, offset_t osc, offset_t isc,
                          const reduce_t * absolute)
    {
        for (offset_t c = 0; c < nc; ++c, inp += isc, out += osc)
            *out += static_cast<scalar_t>( (*absolute++) * (*inp) );
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void diag_absolute(scalar_t * out, offset_t nc, offset_t osc,
                       const reduce_t * absolute)
    {
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out += static_cast<scalar_t>( *absolute++ );
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_membrane(scalar_t * out, const scalar_t * inp,
                          offset_t x, offset_t nx, offset_t sx,
                          offset_t y, offset_t ny, offset_t sy,
                          offset_t z, offset_t nz, offset_t sz,
                          offset_t nc, offset_t osc, offset_t isc,
                          const reduce_t * absolute,
                          const reduce_t * membrane,
                          reduce_t m100, reduce_t m010, reduce_t m001)
    {
        /* NOTE:
         *      m100 = -lx
         *      m010 = -ly
         *      m001 = -lz
         *
         * where lx = 1/(vx[0]*vx[0])
         *       ly = 1/(vx[1]*vx[1])
         *       lz = 1/(vx[2]*vx[2])
         */

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        signed char fy0 = bound_utils_y::sign(y-1, ny);
        signed char fy1 = bound_utils_y::sign(y+1, ny);
        signed char fz0 = bound_utils_z::sign(z-1, nz);
        signed char fz1 = bound_utils_z::sign(z+1, nz);
        offset_t    x0 = (bound_utils_x::index(x-1, nx) - x) * sx;
        offset_t    x1 = (bound_utils_x::index(x+1, nx) - x) * sx;
        offset_t    y0 = (bound_utils_y::index(y-1, ny) - y) * sy;
        offset_t    y1 = (bound_utils_y::index(y+1, ny) - y) * sy;
        offset_t    z0 = (bound_utils_z::index(z-1, nz) - z) * sz;
        offset_t    z1 = (bound_utils_z::index(z+1, nz) - z) * sz;

        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc)
        {
            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [center, inp](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            reduce_t o000 = (*membrane++) * (
                m100*(get(x0, fx0) + get(x1, fx1)) +
                m010*(get(y0, fy0) + get(y1, fy1)) +
                m001*(get(z0, fz0) + get(z1, fz1)));
            if (absolute)
                o000 += (*absolute++) * center;
            *out += static_cast<scalar_t>(o000);
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void diag_membrane(scalar_t * out, offset_t nc, offset_t osc,
                       const reduce_t * absolute,
                       const reduce_t * membrane,
                       reduce_t m100, reduce_t m010, reduce_t m001)
    {
        /* NOTE:
         *      m100 = -lx
         *      m010 = -ly
         *      m001 = -lz
         *
         * where lx = 1/(vx[0]*vx[0])
         *       ly = 1/(vx[1]*vx[1])
         *       lz = 1/(vx[2]*vx[2])
         */
        for (offset_t c = 0; c < n; ++c, out += osc)
        {
            reduce_t o000 = -2 * (*membrane++) * (m100 + m010 + m001);
            if (absolute)
                o000 += (*absolute++);
            *out += static_cast<scalar_t>(o000);
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_bending(scalar_t * out, const scalar_t * inp,
                         offset_t x, offset_t nx, offset_t sx,
                         offset_t y, offset_t ny, offset_t sy,
                         offset_t z, offset_t nz, offset_t sz,
                         offset_t nc, offset_t osc, offset_t isc,
                         const reduce_t * absolute,
                         const reduce_t * membrane,
                         const reduce_t * bending,
                         reduce_t m100, reduce_t m010, reduce_t m001,
                         reduce_t b100, reduce_t b010, reduce_t b001,
                         reduce_t b200, reduce_t b020, reduce_t b020,
                         reduce_t b110, reduce_t b101, reduce_t b011)
    {
        /* NOTE:
         *      m100 = -lx
         *      m010 = -ly
         *      m001 = -lz
         *      b100 = -4 * lx * (lx + ly + lz)
         *      b010 = -4 * ly * (lx + ly + lz)
         *      b001 = -4 * lz * (lx + ly + lz)
         *      b200 = lx * lx
         *      b020 = ly * ly
         *      b002 = lz * lz
         *      b110 = 2 * lx * ly
         *      b101 = 2 * lx * lz
         *      b011 = 2 * ly * lz
         *
         * where lx = 1/(vx[0]*vx[0])
         *       ly = 1/(vx[1]*vx[1])
         *       lz = 1/(vx[2]*vx[2])
         */

        signed char fx00 = bound_utils_x::sign(x-2, nx);
        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        signed char fx11 = bound_utils_x::sign(x+2, nx);
        signed char fy00 = bound_utils_y::sign(y-2, ny);
        signed char fy0  = bound_utils_y::sign(y-1, ny);
        signed char fy1  = bound_utils_y::sign(y+1, ny);
        signed char fy11 = bound_utils_y::sign(y+2, ny);
        signed char fz00 = bound_utils_z::sign(z-2, nz);
        signed char fz0  = bound_utils_z::sign(z-1, nz);
        signed char fz1  = bound_utils_z::sign(z+1, nz);
        signed char fz11 = bound_utils_z::sign(z+2, nz);
        offset_t    x00 = (bound_utils_x::index(x-2, nx) - x) * sx;
        offset_t    x0  = (bound_utils_x::index(x-1, nx) - x) * sx;
        offset_t    x1  = (bound_utils_x::index(x+1, nx) - x) * sx;
        offset_t    x11 = (bound_utils_x::index(x+2, nx) - x) * sx;
        offset_t    y00 = (bound_utils_y::index(y-2, ny) - y) * sy;
        offset_t    y0  = (bound_utils_y::index(y-1, ny) - y) * sy;
        offset_t    y1  = (bound_utils_y::index(y+1, ny) - y) * sy;
        offset_t    y11 = (bound_utils_y::index(y+2, ny) - y) * sy;
        offset_t    z00 = (bound_utils_z::index(z-2, nz) - z) * sz;
        offset_t    z0  = (bound_utils_z::index(z-1, nz) - z) * sz;
        offset_t    z1  = (bound_utils_z::index(z+1, nz) - z) * sz;
        offset_t    z11 = (bound_utils_z::index(z+2, nz) - z) * sz;

        reduce_t aa, mm, bb, w100, w200, w110, w101, w011;

        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc)
        {
            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [center, inp](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            aa = absolute ? *(absolute++) : static_cast<reduce_t>(0);
            mm = membrane ? *(membrane++) : static_cast<reduce_t>(0);
            bb = bending  ? *(bending++)  : static_cast<reduce_t>(0);
            w100 = bb * b100 + mm * m100;
            w010 = bb * b010 + mm * m010;
            w001 = bb * b001 + mm * m001;
            w200 = bb * b200;
            w020 = bb * b020;
            w002 = bb * b002;
            w110 = bb * b110;
            w101 = bb * b101;
            w011 = bb * b011;

            *out += static_cast<scalar_t>(
                  aa   * center
                + w100 * (get(x0, fx0) + get(x1, fx1))
                + w010 * (get(y0, fy0) + get(y1, fy1))
                + w001 * (get(z0, fz0) + get(z1, fz1))
                + w200 * (get(x00, fx00) + get(x11, fx11))
                + w020 * (get(y00, fy00) + get(y11, fy11))
                + w002 * (get(z00, fz00) + get(z11, fz11))
                + w110 * (get(x0+y0, fx0*fy0) + get(x1+y0, fx1*fy0) +
                          get(x1+y0, fx1*fy0) + get(x1+y1, fx1*fy1))
                + w101 * (get(x0+z0, fx0*fz0) + get(x1+z0, fx1*fz0) +
                          get(x1+z0, fx1*fz0) + get(x1+z1, fx1*fz1))
                + w011 * (get(y0+z0, fy0*fz0) + get(y1+z0, fy1*fz0) +
                          get(y1+z0, fy1*fz0) + get(y1+z1, fy1*fz1))
            );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void diag_bending(scalar_t * out, offset_t nc, offset_t osc, o
                      const reduce_t * absolute,
                      const reduce_t * membrane,
                      const reduce_t * bending,
                      reduce_t m100, reduce_t m010, reduce_t m001,
                      reduce_t b100, reduce_t b010, reduce_t b001,
                      reduce_t b200, reduce_t b020, reduce_t b020,
                      reduce_t b110, reduce_t b101, reduce_t b011)
    {
        /* NOTE:
         *      m100 = -lx
         *      m010 = -ly
         *      m001 = -lz
         *      b100 = -4 * lx * (lx + ly + lz)
         *      b010 = -4 * ly * (lx + ly + lz)
         *      b001 = -4 * lz * (lx + ly + lz)
         *      b200 = lx * lx
         *      b020 = ly * ly
         *      b002 = lz * lz
         *      b110 = 2 * lx * ly
         *      b101 = 2 * lx * lz
         *      b011 = 2 * ly * lz
         *
         * where lx = 1/(vx[0]*vx[0])
         *       ly = 1/(vx[1]*vx[1])
         *       lz = 1/(vx[2]*vx[2])
         */
        const reduce_t *a = absolute, *m = membrane, *b = bending;
        reduce_t aa, mm, bb, w100, w200, w110, w101, w011;

        for (offset_t c = 0; c < n; ++c, out += osc)
        {
            bb = *bending++;
            w100 = bb * b100;
            w010 = bb * b010;
            w001 = bb * b001;
            w200 = bb * b200;
            w020 = bb * b020;
            w002 = bb * b002;
            w110 = bb * b110;
            w101 = bb * b101;
            w011 = bb * b011;
            if (membrane)
            {
                mm = *membrane++;
                w100 += mm * m100;
                w010 += mm * m010;
                w001 += mm * m001;
            }

            reduce_t o000 = - 2 * (w100 + w010 + w001 + w200 + w020)
                            - 4 * (w110 + w101 + w011);
            if (absolute)
                o000 += (*absolute++);

            *out += static_cast<scalar_t>(o000);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_absolute_rls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t nc, offset_t osc, offset_t isc, offset_t wsc,
        const reduce_t * absolute)
    {
        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc, wgt += wsc)
        {
            *out += static_cast<scalar_t>(
                (*absolute++) * static_cast<reduce_t>(*wgt) * static_cast<reduce_t>(*inp)
            );
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    void diag_absolute_rls(
        scalar_t * out, const scalar_t * wgt,
        offset_t nc, offset_t osc, offset_t wsc,
        const reduce_t * absolute)
    {
        for (offset_t c = 0; c < n; ++c, out += osc, wgt += wsc)
        {
            *out += static_cast<scalar_t>(
                (*absolute++) * static_cast<reduce_t>(*wgt)
            );
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_absolute_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t nc, offset_t osc, offset_t isc, offset_t wsc,
        const reduce_t * absolute)
    {
        reduce_t w000 = static_cast<reduce_t>(*wgt);
        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc, wgt += wsc)
        {
            *out += static_cast<scalar_t>(
                (*absolute++) * w000 * static_cast<reduce_t>(*inp)
            );
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    void diag_absolute_jrls(
        scalar_t * out, const scalar_t * wgt,
        offset_t nc, offset_t osc, offset_t wsc,
        const reduce_t * absolute)
    {
        reduce_t w000 = static_cast<reduce_t>(*wgt);
        for (offset_t c = 0; c < n; ++c, out += osc, wgt += wsc)
        {
            *out += static_cast<scalar_t>( (*absolute++) * w000 );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_membrane_rls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t x, offset_t nx, offset_t isx, offset_t wsx,
        offset_t y, offset_t ny, offset_t isy, offset_t wsy,
        offset_t z, offset_t nz, offset_t isz, offset_t wsz,
        offset_t nc, offset_t osc, offset_t isc, offset_t wsc,
        const reduce_t * membrane, reduce_t m100, reduce_t m010, reduce_t m001)
    {
        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        signed char fy0 = bound_utils_y::sign(y-1, ny);
        signed char fy1 = bound_utils_y::sign(y+1, ny);
        signed char fz0 = bound_utils_z::sign(z-1, nz);
        signed char fz1 = bound_utils_z::sign(z+1, nz);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x);
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x);
        offset_t    iy0 = (bound_utils_y::index(y-1, ny) - y);
        offset_t    iy1 = (bound_utils_y::index(y+1, ny) - y);
        offset_t    iz0 = (bound_utils_z::index(z-1, nz) - z);
        offset_t    iz1 = (bound_utils_z::index(z+1, nz) - z);
        offset_t    wx0 = ix0 * wsx;
        offset_t    wx1 = ix1 * wsx;
        offset_t    wy0 = iy0 * wsy;
        offset_t    wy1 = iy1 * wsy;
        offset_t    wz0 = iz0 * wsz;
        offset_t    wz1 = iz1 * wsz;
        ix0 *= isx;
        ix1 *= isx;
        iy0 *= isy;
        iy1 *= isy;
        iz0 *= isz;
        iz1 *= isz;

        const reduce_t *m = membrane;

        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc, wgt += wsc)
        {

            reduce_t wcenter = static_cast<reduce_t>(*wgt);
            auto wget = [wcenter, wgt](offset_t o)
            {
                return bound::cget<reduce_t>(wgt, o) + wcenter;
            };

            reduce_t w1m00 = m100 * wget(wx0);
            reduce_t w1p00 = m100 * wget(wx1);
            reduce_t w01m0 = m010 * wget(wy0);
            reduce_t w01p0 = m010 * wget(wy1);
            reduce_t w001m = m001 * wget(wz0);
            reduce_t w001p = m001 * wget(wz1);

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [center, inp](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            *out += static_cast<scalar_t>(
                (*m++) * 0.5 * (w1m00*get(ix0, fx0) +  w1p00*get(ix1, fx1) +
                                w01m0*get(iy0, fy0) +  w01p0*get(iy1, fy1) +
                                w001m*get(iz0, fz0) +  w001p*get(iz1, fz1))
            );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void diag_membrane_rls(
        scalar_t * out, const scalar_t * wgt,
        offset_t x, offset_t nx, offset_t wsx,
        offset_t y, offset_t ny, offset_t wsy,
        offset_t z, offset_t nz, offset_t wsz,
        offset_t nc, offset_t osc, offset_t wsc,
        const reduce_t * membrane, reduce_t m100, reduce_t m010, reduce_t m001)
    {
        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        signed char fy0 = bound_utils_y::sign(y-1, ny);
        signed char fy1 = bound_utils_y::sign(y+1, ny);
        signed char fz0 = bound_utils_z::sign(z-1, nz);
        signed char fz1 = bound_utils_z::sign(z+1, nz);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x) * wsx;
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x) * wsx;
        offset_t    iy0 = (bound_utils_y::index(y-1, ny) - y) * wsy;
        offset_t    iy1 = (bound_utils_y::index(y+1, ny) - y) * wsy;
        offset_t    iz0 = (bound_utils_z::index(z-1, nz) - z) * wsz;
        offset_t    iz1 = (bound_utils_z::index(z+1, nz) - z) * wsz;

        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc, wgt += wsc)
        {

            reduce_t wcenter = static_cast<reduce_t>(*wgt);
            auto wget = [wcenter, wgt](offset_t o)
            {
                return bound::cget<reduce_t>(wgt, o) + wcenter;
            };

            reduce_t w1m00 = m100 * wget(ix0);
            reduce_t w1p00 = m100 * wget(ix1);
            reduce_t w01m0 = m010 * wget(iy0);
            reduce_t w01p0 = m010 * wget(iy1);
            reduce_t w001m = m001 * wget(iz0);
            reduce_t w001p = m001 * wget(iz1);

            *out += static_cast<scalar_t>(
                - (*membrane++) * 0.5 * (w1m00*fx0 +  w1p00*fx1 +
                                         w01m0*fy0 +  w01p0*fy1 +
                                         w001m*fz0 +  w001p*fz1)
            );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_membrane_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t x, offset_t nx, offset_t isx, offset_t wsx,
        offset_t y, offset_t ny, offset_t isy, offset_t wsy,
        offset_t z, offset_t nz, offset_t isz, offset_t wsz,
        offset_t nc, offset_t osc, offset_t isc,
        const reduce_t * membrane, reduce_t m100, reduce_t m010, reduce_t m001)
    {
        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        signed char fy0 = bound_utils_y::sign(y-1, ny);
        signed char fy1 = bound_utils_y::sign(y+1, ny);
        signed char fz0 = bound_utils_z::sign(z-1, nz);
        signed char fz1 = bound_utils_z::sign(z+1, nz);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x);
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x);
        offset_t    iy0 = (bound_utils_y::index(y-1, ny) - y);
        offset_t    iy1 = (bound_utils_y::index(y+1, ny) - y);
        offset_t    iz0 = (bound_utils_z::index(z-1, nz) - z);
        offset_t    iz1 = (bound_utils_z::index(z+1, nz) - z);
        offset_t    wx0 = ix0 * wsx;
        offset_t    wx1 = ix1 * wsx;
        offset_t    wy0 = iy0 * wsy;
        offset_t    wy1 = iy1 * wsy;
        offset_t    wz0 = iz0 * wsz;
        offset_t    wz1 = iz1 * wsz;
        ix0 *= isx;
        ix1 *= isx;
        iy0 *= isy;
        iy1 *= isy;
        iz0 *= isz;
        iz1 *= isz;

        const reduce_t *m = membrane;

        reduce_t wcenter = static_cast<reduce_t>(*wgt);
        auto wget = [wcenter, wgt](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(wgt, o) + wcenter;
        };

        reduce_t w1m00 = m100 * wget(wx0);
        reduce_t w1p00 = m100 * wget(wx1);
        reduce_t w01m0 = m010 * wget(wy0);
        reduce_t w01p0 = m010 * wget(wy1);
        reduce_t w001m = m001 * wget(wz0);
        reduce_t w001p = m001 * wget(wz1);

        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc)
        {
            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [center, inp](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            *out += static_cast<scalar_t>(
                (*m++) * 0.5 * (w1m00*get(ix0, fx0) +  w1p00*get(ix1, fx1) +
                                w01m0*get(iy0, fy0) +  w01p0*get(iy1, fy1) +
                                w001m*get(iz0, fz0) +  w001p*get(iz1, fz1))
            );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void diag_membrane_jrls(
        scalar_t * out, const scalar_t * wgt,
        offset_t x, offset_t nx, offset_t wsx,
        offset_t y, offset_t ny, offset_t wsy,
        offset_t z, offset_t nz, offset_t wsz,
        offset_t nc, offset_t osc,
        const reduce_t * membrane, reduce_t m100, reduce_t m010, reduce_t m001)
    {
        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        signed char fy0 = bound_utils_y::sign(y-1, ny);
        signed char fy1 = bound_utils_y::sign(y+1, ny);
        signed char fz0 = bound_utils_z::sign(z-1, nz);
        signed char fz1 = bound_utils_z::sign(z+1, nz);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x) * wsx;
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x) * wsx;
        offset_t    iy0 = (bound_utils_y::index(y-1, ny) - y) * wsy;
        offset_t    iy1 = (bound_utils_y::index(y+1, ny) - y) * wsy;
        offset_t    iz0 = (bound_utils_z::index(z-1, nz) - z) * wsz;
        offset_t    iz1 = (bound_utils_z::index(z+1, nz) - z) * wsz;

        reduce_t wcenter = static_cast<reduce_t>(*wgt);
        auto wget = [wcenter, wgt](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o) + wcenter;
        };

        reduce_t w1m00 = m100 * wget(wx0);
        reduce_t w1p00 = m100 * wget(wx1);
        reduce_t w01m0 = m010 * wget(wy0);
        reduce_t w01p0 = m010 * wget(wy1);
        reduce_t w001m = m001 * wget(wz0);
        reduce_t w001p = m001 * wget(wz1);

        for (offset_t c = 0; c < n; ++c, out += osc)
        {

            *out += static_cast<scalar_t>(
                - (*membrane++) * 0.5 * (w1m00*fx0 +  w1p00*fx1 +
                                         w01m0*fy0 +  w01p0*fy1 +
                                         w001m*fz0 +  w001p*fz1)
            );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_bending_rls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t x, offset_t nx, offset_t isx, offset_t wsx,
        offset_t y, offset_t ny, offset_t isy, offset_t wsy,
        offset_t z, offset_t nz, offset_t isz, offset_t wsz,
        offset_t nc, offset_t osc, offset_t isc, offset_t wsc,
        const reduce_t * bending,
        reduce_t b100, reduce_t b010, reduce_t b001,
        reduce_t b200, reduce_t b020, reduce_t b002,
        reduce_t b110, reduce_t b101, reduce_t b011)
    {
        /* NOTE:
         *      b100 = -4 * lx * (lx + ly + lz)
         *      b010 = -4 * ly * (lx + ly + lz)
         *      b001 = -4 * lz * (lx + ly + lz)
         *      b200 = lx * lx
         *      b020 = ly * ly
         *      b002 = lz * lz
         *      b110 = 2 * lx * ly
         *      b101 = 2 * lx * lz
         *      b011 = 2 * ly * lz
         *
         * where lx = 1/(vx[0]*vx[0])
         *       ly = 1/(vx[1]*vx[1])
         *       lz = 1/(vx[2]*vx[2])
         */

        signed char fx0 = bound_utils_x::sign(x-2, nx);
        signed char fx1 = bound_utils_x::sign(x-1, nx);
        signed char fx3 = bound_utils_x::sign(x+1, nx);
        signed char fx4 = bound_utils_x::sign(x+2, nx);
        signed char fy0 = bound_utils_y::sign(y-2, ny);
        signed char fy1 = bound_utils_y::sign(y-1, ny);
        signed char fy3 = bound_utils_y::sign(y+1, ny);
        signed char fy4 = bound_utils_y::sign(y+2, ny);
        signed char fz0 = bound_utils_z::sign(z-2, nz);
        signed char fz1 = bound_utils_z::sign(z-1, nz);
        signed char fz3 = bound_utils_z::sign(z+1, nz);
        signed char fz4 = bound_utils_z::sign(z+2, nz);
        offset_t    ix0 = (bound_utils_x::index(x-2, nx) - x);
        offset_t    ix1 = (bound_utils_x::index(x-1, nx) - x);
        offset_t    ix3 = (bound_utils_x::index(x+1, nx) - x);
        offset_t    ix4 = (bound_utils_x::index(x+2, nx) - x);
        offset_t    iy0 = (bound_utils_y::index(y-2, ny) - y);
        offset_t    iy1 = (bound_utils_y::index(y-1, ny) - y);
        offset_t    iy3 = (bound_utils_y::index(y+1, ny) - y);
        offset_t    iy4 = (bound_utils_y::index(y+2, ny) - y);
        offset_t    iz0 = (bound_utils_z::index(z-2, nz) - z);
        offset_t    iz1 = (bound_utils_z::index(z-1, nz) - z);
        offset_t    iz3 = (bound_utils_z::index(z+1, nz) - z);
        offset_t    iz4 = (bound_utils_z::index(z+2, nz) - z);
        offset_t    wx0 = ix0 * wsx;
        offset_t    wx1 = ix1 * wsx;
        offset_t    wx3 = ix3 * wsx;
        offset_t    wx4 = ix4 * wsx;
        offset_t    wy0 = iy0 * wsy;
        offset_t    wy1 = iy1 * wsy;
        offset_t    wy3 = iy3 * wsy;
        offset_t    wy4 = iy4 * wsy;
        offset_t    wz0 = iz0 * wsz;
        offset_t    wz1 = iz1 * wsz;
        offset_t    wz3 = iz3 * wsz;
        offset_t    wz4 = iz4 * wsz;
        ix00 *= isx;
        ix0  *= isx;
        ix1  *= isx;
        ix11 *= isx;
        iy00 *= isy;
        iy0  *= isy;
        iy1  *= isy;
        iy11 *= isy;
        iz00 *= isz;
        iz0  *= isz;
        iz1  *= isz;
        iz11 *= isz;

        const reduce_t *b = bending;

        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc, wgt += wsc)
        {

            reduce_t w222 = static_cast<reduce_t>(*wgt);
            auto wget = [wgt](offset_t o)
            {
                return bound::cget<reduce_t>(wgt, o);
            };

            reduce_t w122 = wget(wx0);
            reduce_t w322 = wget(wx1);
            reduce_t w212 = wget(wy0);
            reduce_t w232 = wget(wy1);
            reduce_t w221 = wget(wz0);
            reduce_t w223 = wget(wz1);

            reduce_t w022 = wget(wx00);
            reduce_t w422 = wget(wx11);
            reduce_t w202 = wget(wy00);
            reduce_t w242 = wget(wy11);
            reduce_t w220 = wget(wz00);
            reduce_t w224 = wget(wz11);

            reduce_t w112 = wget(wx0+wy0);
            reduce_t w132 = wget(wx0+wy1);
            reduce_t w312 = wget(wx1+wy0);
            reduce_t w332 = wget(wx1+wy1);
            reduce_t w121 = wget(wx0+wz0);
            reduce_t w123 = wget(wx0+wz1);
            reduce_t w321 = wget(wx1+wz0);
            reduce_t w323 = wget(wx1+wz1);
            reduce_t w211 = wget(wy0+wz0);
            reduce_t w213 = wget(wy0+wz1);
            reduce_t w231 = wget(wy1+wz0);
            reduce_t w233 = wget(wy1+wz1);

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [center, inp](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            reduce_t m122 = (b100 - 2*b200) * (w222 + w122)
                            - 2*b200 * (w322 + w022)
                            - b110 * (w212 + w112 + w232 + w132)
                            - b101 * (w221 + w121 + w223 + w123);
            reduce_t m322 = (b100 - 2*b200) * (w222 + w322)
                            - 2*b200 * (w422 + w122)
                            - b110 * (w232 + w332 + w212 + w312)
                            - b101 * (w223 + w323 + w221 + w321);

            reduce_t m212 = (b010 - 2*b020) * (w222 + w212)
                            - 2*b020 * (w232 + w202)
                            - b110 * (w122 + w112 + w322 + w132)
                            - b011 * (w221 + w211 + w223 + w213);
            reduce_t m232 = (b010 - 2*b020) * (w222 + w232)
                            - 2*b020 * (w242 + w212)
                            - b110 * (w322 + w332 + w122 + w312)
                            - b011 * (w223 + w233 + w221 + w231);

            reduce_t m221 = (b001 - 2*b002) * (w222 + w221)
                            - 2*b002 * (w223 + w220)
                            - b101 * (w122 + w121 + w322 + w123)
                            - b011 * (w212 + w211 + w232 + w231);
            reduce_t m223 = (b001 - 2*b002) * (w222 + w223)
                            - 2*b002 * (w224 + w221)
                            - b101 * (w322 + w323 + w122 + w321)
                            - b011 * (w232 + w233 + w212 + w213);

            reduce_t m022 = b200 * (2 * w122 + w022 + w222);
            reduce_t m422 = b200 * (2 * w322 + w422 + w222);
            reduce_t m202 = b020 * (2 * w212 + w202 + w222);
            reduce_t m242 = b020 * (2 * w232 + w242 + w222);
            reduce_t m220 = b002 * (2 * w221 + w220 + w222);
            reduce_t m224 = b002 * (2 * w223 + w224 + w222);

            reduce_t m112 = b110 * (w222 + w122 + w212 + w112);
            reduce_t m132 = b110 * (w222 + w122 + w232 + w132);
            reduce_t m312 = b110 * (w222 + w322 + w212 + w312);
            reduce_t m332 = b110 * (w222 + w322 + w232 + w332);

            reduce_t m121 = b101 * (w222 + w122 + w221 + w121);
            reduce_t m123 = b101 * (w222 + w122 + w223 + w123);
            reduce_t m321 = b101 * (w222 + w322 + w221 + w321);
            reduce_t m323 = b101 * (w222 + w322 + w223 + w323);

            reduce_t m211 = b011 * (w222 + w221 + w212 + w112);
            reduce_t m213 = b011 * (w222 + w221 + w232 + w132);
            reduce_t m231 = b011 * (w222 + w223 + w212 + w312);
            reduce_t m233 = b011 * (w222 + w223 + w232 + w332);

            *out += static_cast<scalar_t>(
                (*b++) * 0.25 * ((m122*get(ix1, fx1) +  m322*get(ix3, fx3) +
                                  m212*get(iy1, fy1) +  m232*get(iy3, fy3) +
                                  m221*get(iz1, fz1) +  m223*get(iz3, fz3)) +
                                 (m022*get(ix0, fx0) +  m422*get(ix4, fx4) +
                                  m202*get(iy0, fy0) +  m242*get(iy4, fy4) +
                                  m220*get(iz0, fz0) +  m224*get(iz4, fz4)) +
                                 (m112*get(ix1+iy1, fx1*fy1) +  m132*get(ix1+iy3, fx1*fy3) +
                                  m312*get(ix3+iy1, fx3*fy1) +  m332*get(ix3+iy3, fx3*fy3) +
                                  m121*get(ix1+iz1, fx1*fz1) +  m123*get(ix1+iz3, fx1*fz3) +
                                  m321*get(ix3+iz1, fx3*fz1) +  m323*get(ix3+iz3, fx3*fz3) +
                                  m211*get(iy1+iz1, fy1*fz1) +  m213*get(iy1+iz3, fy1*fz3) +
                                  m231*get(iy3+iz1, fy3*fz1) +  m233*get(iy3+iz3, fy3*fz3)))
            );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void diag_bending_rls(
        scalar_t * out, const scalar_t * wgt,
        offset_t x, offset_t nx, offset_t wsx,
        offset_t y, offset_t ny, offset_t wsy,
        offset_t z, offset_t nz, offset_t wsz,
        offset_t nc, offset_t osc, offset_t wsc,
        const reduce_t * bending,
        reduce_t b100, reduce_t b010, reduce_t b001,
        reduce_t b200, reduce_t b020, reduce_t b002,
        reduce_t b110, reduce_t b101, reduce_t b011)
    {
        /* NOTE:
         *      b100 = -4 * lx * (lx + ly + lz)
         *      b010 = -4 * ly * (lx + ly + lz)
         *      b001 = -4 * lz * (lx + ly + lz)
         *      b200 = lx * lx
         *      b020 = ly * ly
         *      b002 = lz * lz
         *      b110 = 2 * lx * ly
         *      b101 = 2 * lx * lz
         *      b011 = 2 * ly * lz
         *
         * where lx = 1/(vx[0]*vx[0])
         *       ly = 1/(vx[1]*vx[1])
         *       lz = 1/(vx[2]*vx[2])
         */

        signed char fx0 = bound_utils_x::sign(x-2, nx);
        signed char fx1 = bound_utils_x::sign(x-1, nx);
        signed char fx3 = bound_utils_x::sign(x+1, nx);
        signed char fx4 = bound_utils_x::sign(x+2, nx);
        signed char fy0 = bound_utils_y::sign(y-2, ny);
        signed char fy1 = bound_utils_y::sign(y-1, ny);
        signed char fy3 = bound_utils_y::sign(y+1, ny);
        signed char fy4 = bound_utils_y::sign(y+2, ny);
        signed char fz0 = bound_utils_z::sign(z-2, nz);
        signed char fz1 = bound_utils_z::sign(z-1, nz);
        signed char fz3 = bound_utils_z::sign(z+1, nz);
        signed char fz4 = bound_utils_z::sign(z+2, nz);
        offset_t    ix0 = (bound_utils_x::index(x-2, nx) - x) * wsx;
        offset_t    ix1 = (bound_utils_x::index(x-1, nx) - x) * wsx;
        offset_t    ix3 = (bound_utils_x::index(x+1, nx) - x) * wsx;
        offset_t    ix4 = (bound_utils_x::index(x+2, nx) - x) * wsx;
        offset_t    iy0 = (bound_utils_y::index(y-2, ny) - y) * wsy;
        offset_t    iy1 = (bound_utils_y::index(y-1, ny) - y) * wsy;
        offset_t    iy3 = (bound_utils_y::index(y+1, ny) - y) * wsy;
        offset_t    iy4 = (bound_utils_y::index(y+2, ny) - y) * wsy;
        offset_t    iz0 = (bound_utils_z::index(z-2, nz) - z) * wsz;
        offset_t    iz1 = (bound_utils_z::index(z-1, nz) - z) * wsz;
        offset_t    iz3 = (bound_utils_z::index(z+1, nz) - z) * wsz;
        offset_t    iz4 = (bound_utils_z::index(z+2, nz) - z) * wsz;

        for (offset_t c = 0; c < n; ++c, out += osc, wgt += wsc)
        {

            reduce_t w222 = static_cast<reduce_t>(*wgt);
            auto wget = [wgt](offset_t o)
            {
                return bound::cget<reduce_t>(wgt, o);
            };

            reduce_t w122 = wget(wx0);
            reduce_t w322 = wget(wx1);
            reduce_t w212 = wget(wy0);
            reduce_t w232 = wget(wy1);
            reduce_t w221 = wget(wz0);
            reduce_t w223 = wget(wz1);

            reduce_t w022 = wget(wx00);
            reduce_t w422 = wget(wx11);
            reduce_t w202 = wget(wy00);
            reduce_t w242 = wget(wy11);
            reduce_t w220 = wget(wz00);
            reduce_t w224 = wget(wz11);

            reduce_t w112 = wget(wx0+wy0);
            reduce_t w132 = wget(wx0+wy1);
            reduce_t w312 = wget(wx1+wy0);
            reduce_t w332 = wget(wx1+wy1);
            reduce_t w121 = wget(wx0+wz0);
            reduce_t w123 = wget(wx0+wz1);
            reduce_t w321 = wget(wx1+wz0);
            reduce_t w323 = wget(wx1+wz1);
            reduce_t w211 = wget(wy0+wz0);
            reduce_t w213 = wget(wy0+wz1);
            reduce_t w231 = wget(wy1+wz0);
            reduce_t w233 = wget(wy1+wz1);

            reduce_t m122 = (b100 - 2*b200) * (w222 + w122)
                            - 2*b200 * (w322 + w022)
                            - b110 * (w212 + w112 + w232 + w132)
                            - b101 * (w221 + w121 + w223 + w123);
            reduce_t m322 = (b100 - 2*b200) * (w222 + w322)
                            - 2*b200 * (w422 + w122)
                            - b110 * (w232 + w332 + w212 + w312)
                            - b101 * (w223 + w323 + w221 + w321);

            reduce_t m212 = (b010 - 2*b020) * (w222 + w212)
                            - 2*b020 * (w232 + w202)
                            - b110 * (w122 + w112 + w322 + w132)
                            - b011 * (w221 + w211 + w223 + w213);
            reduce_t m232 = (b010 - 2*b020) * (w222 + w232)
                            - 2*b020 * (w242 + w212)
                            - b110 * (w322 + w332 + w122 + w312)
                            - b011 * (w223 + w233 + w221 + w231);

            reduce_t m221 = (b001 - 2*b002) * (w222 + w221)
                            - 2*b002 * (w223 + w220)
                            - b101 * (w122 + w121 + w322 + w123)
                            - b011 * (w212 + w211 + w232 + w231);
            reduce_t m223 = (b001 - 2*b002) * (w222 + w223)
                            - 2*b002 * (w224 + w221)
                            - b101 * (w322 + w323 + w122 + w321)
                            - b011 * (w232 + w233 + w212 + w213);

            reduce_t m022 = b200 * (2 * w122 + w022 + w222);
            reduce_t m422 = b200 * (2 * w322 + w422 + w222);
            reduce_t m202 = b020 * (2 * w212 + w202 + w222);
            reduce_t m242 = b020 * (2 * w232 + w242 + w222);
            reduce_t m220 = b002 * (2 * w221 + w220 + w222);
            reduce_t m224 = b002 * (2 * w223 + w224 + w222);

            reduce_t m112 = b110 * (w222 + w122 + w212 + w112);
            reduce_t m132 = b110 * (w222 + w122 + w232 + w132);
            reduce_t m312 = b110 * (w222 + w322 + w212 + w312);
            reduce_t m332 = b110 * (w222 + w322 + w232 + w332);

            reduce_t m121 = b101 * (w222 + w122 + w221 + w121);
            reduce_t m123 = b101 * (w222 + w122 + w223 + w123);
            reduce_t m321 = b101 * (w222 + w322 + w221 + w321);
            reduce_t m323 = b101 * (w222 + w322 + w223 + w323);

            reduce_t m211 = b011 * (w222 + w221 + w212 + w112);
            reduce_t m213 = b011 * (w222 + w221 + w232 + w132);
            reduce_t m231 = b011 * (w222 + w223 + w212 + w312);
            reduce_t m233 = b011 * (w222 + w223 + w232 + w332);

            *out += static_cast<scalar_t>(
                - (*bending++) * 0.25 * ((m122*fx1 +  m322*fx3 +
                                          m212*fy1 +  m232*fy3 +
                                          m221*fz1 +  m223*fz3) +
                                         (m022*fx0 +  m422*fx4 +
                                          m202*fy0 +  m242*fy4 +
                                          m220*fz0 +  m224*fz4) +
                                         (m112*(fx1*fy1) +  m132*(fx1*fy3) +
                                          m312*(fx3*fy1) +  m332*(fx3*fy3) +
                                          m121*(fx1*fz1) +  m123*(fx1*fz3) +
                                          m321*(fx3*fz1) +  m323*(fx3*fz3) +
                                          m211*(fy1*fz1) +  m213*(fy1*fz3) +
                                          m231*(fy3*fz1) +  m233*(fy3*fz3)))
            );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void vel2mom_bending_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t x, offset_t nx, offset_t isx, offset_t wsx,
        offset_t y, offset_t ny, offset_t isy, offset_t wsy,
        offset_t z, offset_t nz, offset_t isz, offset_t wsz,
        offset_t nc, offset_t osc, offset_t isc,
        const reduce_t * bending,
        reduce_t b100, reduce_t b010, reduce_t b001,
        reduce_t b200, reduce_t b020, reduce_t b002,
        reduce_t b110, reduce_t b101, reduce_t b011)
    {
        /* NOTE:
         *      b100 = -4 * lx * (lx + ly + lz)
         *      b010 = -4 * ly * (lx + ly + lz)
         *      b001 = -4 * lz * (lx + ly + lz)
         *      b200 = lx * lx
         *      b020 = ly * ly
         *      b002 = lz * lz
         *      b110 = 2 * lx * ly
         *      b101 = 2 * lx * lz
         *      b011 = 2 * ly * lz
         *
         * where lx = 1/(vx[0]*vx[0])
         *       ly = 1/(vx[1]*vx[1])
         *       lz = 1/(vx[2]*vx[2])
         */

        signed char fx0 = bound_utils_x::sign(x-2, nx);
        signed char fx1 = bound_utils_x::sign(x-1, nx);
        signed char fx3 = bound_utils_x::sign(x+1, nx);
        signed char fx4 = bound_utils_x::sign(x+2, nx);
        signed char fy0 = bound_utils_y::sign(y-2, ny);
        signed char fy1 = bound_utils_y::sign(y-1, ny);
        signed char fy3 = bound_utils_y::sign(y+1, ny);
        signed char fy4 = bound_utils_y::sign(y+2, ny);
        signed char fz0 = bound_utils_z::sign(z-2, nz);
        signed char fz1 = bound_utils_z::sign(z-1, nz);
        signed char fz3 = bound_utils_z::sign(z+1, nz);
        signed char fz4 = bound_utils_z::sign(z+2, nz);
        offset_t    ix0 = (bound_utils_x::index(x-2, nx) - x);
        offset_t    ix1 = (bound_utils_x::index(x-1, nx) - x);
        offset_t    ix3 = (bound_utils_x::index(x+1, nx) - x);
        offset_t    ix4 = (bound_utils_x::index(x+2, nx) - x);
        offset_t    iy0 = (bound_utils_y::index(y-2, ny) - y);
        offset_t    iy1 = (bound_utils_y::index(y-1, ny) - y);
        offset_t    iy3 = (bound_utils_y::index(y+1, ny) - y);
        offset_t    iy4 = (bound_utils_y::index(y+2, ny) - y);
        offset_t    iz0 = (bound_utils_z::index(z-2, nz) - z);
        offset_t    iz1 = (bound_utils_z::index(z-1, nz) - z);
        offset_t    iz3 = (bound_utils_z::index(z+1, nz) - z);
        offset_t    iz4 = (bound_utils_z::index(z+2, nz) - z);
        offset_t    wx0 = ix0 * wsx;
        offset_t    wx1 = ix1 * wsx;
        offset_t    wx3 = ix3 * wsx;
        offset_t    wx4 = ix4 * wsx;
        offset_t    wy0 = iy0 * wsy;
        offset_t    wy1 = iy1 * wsy;
        offset_t    wy3 = iy3 * wsy;
        offset_t    wy4 = iy4 * wsy;
        offset_t    wz0 = iz0 * wsz;
        offset_t    wz1 = iz1 * wsz;
        offset_t    wz3 = iz3 * wsz;
        offset_t    wz4 = iz4 * wsz;
        ix00 *= isx;
        ix0  *= isx;
        ix1  *= isx;
        ix11 *= isx;
        iy00 *= isy;
        iy0  *= isy;
        iy1  *= isy;
        iy11 *= isy;
        iz00 *= isz;
        iz0  *= isz;
        iz1  *= isz;
        iz11 *= isz;

        const reduce_t *b = bending;

        reduce_t w222 = static_cast<reduce_t>(*wgt);
        auto wget = [wgt](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o);
        };

        reduce_t w122 = wget(wx0);
        reduce_t w322 = wget(wx1);
        reduce_t w212 = wget(wy0);
        reduce_t w232 = wget(wy1);
        reduce_t w221 = wget(wz0);
        reduce_t w223 = wget(wz1);

        reduce_t w022 = wget(wx00);
        reduce_t w422 = wget(wx11);
        reduce_t w202 = wget(wy00);
        reduce_t w242 = wget(wy11);
        reduce_t w220 = wget(wz00);
        reduce_t w224 = wget(wz11);

        reduce_t w112 = wget(wx0+wy0);
        reduce_t w132 = wget(wx0+wy1);
        reduce_t w312 = wget(wx1+wy0);
        reduce_t w332 = wget(wx1+wy1);
        reduce_t w121 = wget(wx0+wz0);
        reduce_t w123 = wget(wx0+wz1);
        reduce_t w321 = wget(wx1+wz0);
        reduce_t w323 = wget(wx1+wz1);
        reduce_t w211 = wget(wy0+wz0);
        reduce_t w213 = wget(wy0+wz1);
        reduce_t w231 = wget(wy1+wz0);
        reduce_t w233 = wget(wy1+wz1);

        reduce_t m122 = (b100 - 2*b200) * (w222 + w122)
                        - 2*b200 * (w322 + w022)
                        - b110 * (w212 + w112 + w232 + w132)
                        - b101 * (w221 + w121 + w223 + w123);
        reduce_t m322 = (b100 - 2*b200) * (w222 + w322)
                        - 2*b200 * (w422 + w122)
                        - b110 * (w232 + w332 + w212 + w312)
                        - b101 * (w223 + w323 + w221 + w321);

        reduce_t m212 = (b010 - 2*b020) * (w222 + w212)
                        - 2*b020 * (w232 + w202)
                        - b110 * (w122 + w112 + w322 + w132)
                        - b011 * (w221 + w211 + w223 + w213);
        reduce_t m232 = (b010 - 2*b020) * (w222 + w232)
                        - 2*b020 * (w242 + w212)
                        - b110 * (w322 + w332 + w122 + w312)
                        - b011 * (w223 + w233 + w221 + w231);

        reduce_t m221 = (b001 - 2*b002) * (w222 + w221)
                        - 2*b002 * (w223 + w220)
                        - b101 * (w122 + w121 + w322 + w123)
                        - b011 * (w212 + w211 + w232 + w231);
        reduce_t m223 = (b001 - 2*b002) * (w222 + w223)
                        - 2*b002 * (w224 + w221)
                        - b101 * (w322 + w323 + w122 + w321)
                        - b011 * (w232 + w233 + w212 + w213);

        reduce_t m022 = b200 * (2 * w122 + w022 + w222);
        reduce_t m422 = b200 * (2 * w322 + w422 + w222);
        reduce_t m202 = b020 * (2 * w212 + w202 + w222);
        reduce_t m242 = b020 * (2 * w232 + w242 + w222);
        reduce_t m220 = b002 * (2 * w221 + w220 + w222);
        reduce_t m224 = b002 * (2 * w223 + w224 + w222);

        reduce_t m112 = b110 * (w222 + w122 + w212 + w112);
        reduce_t m132 = b110 * (w222 + w122 + w232 + w132);
        reduce_t m312 = b110 * (w222 + w322 + w212 + w312);
        reduce_t m332 = b110 * (w222 + w322 + w232 + w332);

        reduce_t m121 = b101 * (w222 + w122 + w221 + w121);
        reduce_t m123 = b101 * (w222 + w122 + w223 + w123);
        reduce_t m321 = b101 * (w222 + w322 + w221 + w321);
        reduce_t m323 = b101 * (w222 + w322 + w223 + w323);

        reduce_t m211 = b011 * (w222 + w221 + w212 + w112);
        reduce_t m213 = b011 * (w222 + w221 + w232 + w132);
        reduce_t m231 = b011 * (w222 + w223 + w212 + w312);
        reduce_t m233 = b011 * (w222 + w223 + w232 + w332);

        for (offset_t c = 0; c < n; ++c, inp += isc, out += osc)
        {
            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [center, inp](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            *out += static_cast<scalar_t>(
                (*b++) * 0.25 * ((m122*get(ix1, fx1) +  m322*get(ix3, fx3) +
                                  m212*get(iy1, fy1) +  m232*get(iy3, fy3) +
                                  m221*get(iz1, fz1) +  m223*get(iz3, fz3)) +
                                 (m022*get(ix0, fx0) +  m422*get(ix4, fx4) +
                                  m202*get(iy0, fy0) +  m242*get(iy4, fy4) +
                                  m220*get(iz0, fz0) +  m224*get(iz4, fz4)) +
                                 (m112*get(ix1+iy1, fx1*fy1) +  m132*get(ix1+iy3, fx1*fy3) +
                                  m312*get(ix3+iy1, fx3*fy1) +  m332*get(ix3+iy3, fx3*fy3) +
                                  m121*get(ix1+iz1, fx1*fz1) +  m123*get(ix1+iz3, fx1*fz3) +
                                  m321*get(ix3+iz1, fx3*fz1) +  m323*get(ix3+iz3, fx3*fz3) +
                                  m211*get(iy1+iz1, fy1*fz1) +  m213*get(iy1+iz3, fy1*fz3) +
                                  m231*get(iy3+iz1, fy3*fz1) +  m233*get(iy3+iz3, fy3*fz3)))
            );
        }
    }

    template <typename  reduce_t, typename scalar_t, typename offset_t>
    void diag_bending_jrls(
        scalar_t * out, const scalar_t * wgt,
        offset_t x, offset_t nx, offset_t wsx,
        offset_t y, offset_t ny, offset_t wsy,
        offset_t z, offset_t nz, offset_t wsz,
        offset_t nc, offset_t osc,
        const reduce_t * bending,
        reduce_t b100, reduce_t b010, reduce_t b001,
        reduce_t b200, reduce_t b020, reduce_t b002,
        reduce_t b110, reduce_t b101, reduce_t b011)
    {
        /* NOTE:
         *      b100 = -4 * lx * (lx + ly + lz)
         *      b010 = -4 * ly * (lx + ly + lz)
         *      b001 = -4 * lz * (lx + ly + lz)
         *      b200 = lx * lx
         *      b020 = ly * ly
         *      b002 = lz * lz
         *      b110 = 2 * lx * ly
         *      b101 = 2 * lx * lz
         *      b011 = 2 * ly * lz
         *
         * where lx = 1/(vx[0]*vx[0])
         *       ly = 1/(vx[1]*vx[1])
         *       lz = 1/(vx[2]*vx[2])
         */

        signed char fx0 = bound_utils_x::sign(x-2, nx);
        signed char fx1 = bound_utils_x::sign(x-1, nx);
        signed char fx3 = bound_utils_x::sign(x+1, nx);
        signed char fx4 = bound_utils_x::sign(x+2, nx);
        signed char fy0 = bound_utils_y::sign(y-2, ny);
        signed char fy1 = bound_utils_y::sign(y-1, ny);
        signed char fy3 = bound_utils_y::sign(y+1, ny);
        signed char fy4 = bound_utils_y::sign(y+2, ny);
        signed char fz0 = bound_utils_z::sign(z-2, nz);
        signed char fz1 = bound_utils_z::sign(z-1, nz);
        signed char fz3 = bound_utils_z::sign(z+1, nz);
        signed char fz4 = bound_utils_z::sign(z+2, nz);
        offset_t    ix0 = (bound_utils_x::index(x-2, nx) - x) * wsx;
        offset_t    ix1 = (bound_utils_x::index(x-1, nx) - x) * wsx;
        offset_t    ix3 = (bound_utils_x::index(x+1, nx) - x) * wsx;
        offset_t    ix4 = (bound_utils_x::index(x+2, nx) - x) * wsx;
        offset_t    iy0 = (bound_utils_y::index(y-2, ny) - y) * wsy;
        offset_t    iy1 = (bound_utils_y::index(y-1, ny) - y) * wsy;
        offset_t    iy3 = (bound_utils_y::index(y+1, ny) - y) * wsy;
        offset_t    iy4 = (bound_utils_y::index(y+2, ny) - y) * wsy;
        offset_t    iz0 = (bound_utils_z::index(z-2, nz) - z) * wsz;
        offset_t    iz1 = (bound_utils_z::index(z-1, nz) - z) * wsz;
        offset_t    iz3 = (bound_utils_z::index(z+1, nz) - z) * wsz;
        offset_t    iz4 = (bound_utils_z::index(z+2, nz) - z) * wsz;

        reduce_t w222 = static_cast<reduce_t>(*wgt);
        auto wget = [wgt](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o);
        };

        reduce_t w122 = wget(wx0);
        reduce_t w322 = wget(wx1);
        reduce_t w212 = wget(wy0);
        reduce_t w232 = wget(wy1);
        reduce_t w221 = wget(wz0);
        reduce_t w223 = wget(wz1);

        reduce_t w022 = wget(wx00);
        reduce_t w422 = wget(wx11);
        reduce_t w202 = wget(wy00);
        reduce_t w242 = wget(wy11);
        reduce_t w220 = wget(wz00);
        reduce_t w224 = wget(wz11);

        reduce_t w112 = wget(wx0+wy0);
        reduce_t w132 = wget(wx0+wy1);
        reduce_t w312 = wget(wx1+wy0);
        reduce_t w332 = wget(wx1+wy1);
        reduce_t w121 = wget(wx0+wz0);
        reduce_t w123 = wget(wx0+wz1);
        reduce_t w321 = wget(wx1+wz0);
        reduce_t w323 = wget(wx1+wz1);
        reduce_t w211 = wget(wy0+wz0);
        reduce_t w213 = wget(wy0+wz1);
        reduce_t w231 = wget(wy1+wz0);
        reduce_t w233 = wget(wy1+wz1);

        reduce_t m122 = (b100 - 2*b200) * (w222 + w122)
                        - 2*b200 * (w322 + w022)
                        - b110 * (w212 + w112 + w232 + w132)
                        - b101 * (w221 + w121 + w223 + w123);
        reduce_t m322 = (b100 - 2*b200) * (w222 + w322)
                        - 2*b200 * (w422 + w122)
                        - b110 * (w232 + w332 + w212 + w312)
                        - b101 * (w223 + w323 + w221 + w321);

        reduce_t m212 = (b010 - 2*b020) * (w222 + w212)
                        - 2*b020 * (w232 + w202)
                        - b110 * (w122 + w112 + w322 + w132)
                        - b011 * (w221 + w211 + w223 + w213);
        reduce_t m232 = (b010 - 2*b020) * (w222 + w232)
                        - 2*b020 * (w242 + w212)
                        - b110 * (w322 + w332 + w122 + w312)
                        - b011 * (w223 + w233 + w221 + w231);

        reduce_t m221 = (b001 - 2*b002) * (w222 + w221)
                        - 2*b002 * (w223 + w220)
                        - b101 * (w122 + w121 + w322 + w123)
                        - b011 * (w212 + w211 + w232 + w231);
        reduce_t m223 = (b001 - 2*b002) * (w222 + w223)
                        - 2*b002 * (w224 + w221)
                        - b101 * (w322 + w323 + w122 + w321)
                        - b011 * (w232 + w233 + w212 + w213);

        reduce_t m022 = b200 * (2 * w122 + w022 + w222);
        reduce_t m422 = b200 * (2 * w322 + w422 + w222);
        reduce_t m202 = b020 * (2 * w212 + w202 + w222);
        reduce_t m242 = b020 * (2 * w232 + w242 + w222);
        reduce_t m220 = b002 * (2 * w221 + w220 + w222);
        reduce_t m224 = b002 * (2 * w223 + w224 + w222);

        reduce_t m112 = b110 * (w222 + w122 + w212 + w112);
        reduce_t m132 = b110 * (w222 + w122 + w232 + w132);
        reduce_t m312 = b110 * (w222 + w322 + w212 + w312);
        reduce_t m332 = b110 * (w222 + w322 + w232 + w332);

        reduce_t m121 = b101 * (w222 + w122 + w221 + w121);
        reduce_t m123 = b101 * (w222 + w122 + w223 + w123);
        reduce_t m321 = b101 * (w222 + w322 + w221 + w321);
        reduce_t m323 = b101 * (w222 + w322 + w223 + w323);

        reduce_t m211 = b011 * (w222 + w221 + w212 + w112);
        reduce_t m213 = b011 * (w222 + w221 + w232 + w132);
        reduce_t m231 = b011 * (w222 + w223 + w212 + w312);
        reduce_t m233 = b011 * (w222 + w223 + w232 + w332);

        reduce_t o000 = -0.25 * ((m122*fx1 +  m322*fx3 +
                                  m212*fy1 +  m232*fy3 +
                                  m221*fz1 +  m223*fz3) +
                                 (m022*fx0 +  m422*fx4 +
                                  m202*fy0 +  m242*fy4 +
                                  m220*fz0 +  m224*fz4) +
                                 (m112*(fx1*fy1) +  m132*(fx1*fy3) +
                                  m312*(fx3*fy1) +  m332*(fx3*fy3) +
                                  m121*(fx1*fz1) +  m123*(fx1*fz3) +
                                  m321*(fx3*fz1) +  m323*(fx3*fz3) +
                                  m211*(fy1*fz1) +  m213*(fy1*fz3) +
                                  m231*(fy3*fz1) +  m233*(fy3*fz3)));

        for (offset_t c = 0; c < n; ++c, out += osc)
        {
            *out += static_cast<scalar_t>( (*bending++) * o000 );
        }
    }
};

} // namespace reg_field
} // namespace jf

#endif // JF_REGULARISERS_1D
