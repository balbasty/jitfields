#ifndef JF_RESTRICT
#define JF_RESTRICT
#include "cuda_switch.h"
#include "spline.h"
#include "bounds.h"

namespace jf {
namespace restrict {

const spline::type Z = spline::type::Nearest;
const spline::type L = spline::type::Linear;
const bound::type B0 = bound::type::NoCheck;
const int zero  = 0;
const int one   = 1;
const int two   = 2;
const int three = 3;

// D - Number of spatial dimensions
// U - Upper bound on the restriction factor
// IX, IY, IZ - Interpolation order
// BX, BY, BZ - Boundary conditions
template <int D, int U=zero,
          spline::type IX=Z,  bound::type BX=B0,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
struct Multiscale {};

/***********************************************************************
 *
 *                                  1D
 *
 **********************************************************************/

 /***                              ANY                              ***/
template <int U, spline::type I, bound::type B>
struct Multiscale<one, U, I, B> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<I>;
    static const int spline_order = static_cast<int>(I);

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  offset_t w, offset_t nw, offset_t sw,
                  reduce_t wscl, reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ixlow = static_cast<offset_t>(ceil(x - 0.5 * (spline_order + 1 ) * wscl));
        offset_t ixupp = static_cast<offset_t>(floor(x + 0.5 * (spline_order + 1 ) * wscl));

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t ix = ixlow; ix <= ixupp; ++ix) {
            reduce_t    dx = spline_utils::weight((x - ix) / wscl);
            signed char sx = bound_utils::sign(ix, nw);
            offset_t    ox = bound_utils::index(ix, nw) * sw;
            acc += static_cast<reduce_t>(bound::get(inp, ox, sx)) * dx;
        }
        *out = static_cast<scalar_t>(acc);
    }
};

 /***                      LINEAR + BOUND 2                         ***/
template <bound::type B>
struct Multiscale<one, two, L, B> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<L>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                   offset_t w, offset_t nw, offset_t sw,
                   reduce_t wscl, reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x));
        offset_t ix0 =  ix1 - 1, ix2 = ix1 + 1, ix3 = ix1 + 2;
        reduce_t dx1 = spline_utils::weight((x - ix1) / wscl);
        reduce_t dx0 = spline_utils::weight((x - ix0) / wscl);
        reduce_t dx2 = spline_utils::weight((ix2 - x) / wscl);
        reduce_t dx3 = spline_utils::weight((ix3 - x) / wscl);
        signed char sx3 = bound_utils::sign(ix3, nw);
        signed char sx2 = bound_utils::sign(ix2, nw);
        signed char sx0 = bound_utils::sign(ix0, nw);
        signed char sx1 = bound_utils::sign(ix1, nw);
        ix3 = bound_utils::index(ix3, nw) * sw;
        ix2 = bound_utils::index(ix2, nw) * sw;
        ix0 = bound_utils::index(ix0, nw) * sw;
        ix1 = bound_utils::index(ix1, nw) * sw;

        *out = static_cast<scalar_t>(
                static_cast<reduce_t>(bound::get(inp, ix0, sx0)) * dx0
              + static_cast<reduce_t>(bound::get(inp, ix1, sx1)) * dx1
              + static_cast<reduce_t>(bound::get(inp, ix2, sx2)) * dx2
              + static_cast<reduce_t>(bound::get(inp, ix3, sx3)) * dx3);
    }
};

/***********************************************************************
 *
 *                                  2D
 *
 **********************************************************************/

 /***                              ANY                              ***/
template <int U,
          spline::type IX, bound::type BX,
          spline::type IY, bound::type BY>
struct Multiscale<two, U, IX, BX, IY, BY> {
    using bound_utils_x = bound::utils<BX>;
    using spline_utils_x = spline::utils<IX>;
    using bound_utils_y = bound::utils<BY>;
    using spline_utils_y = spline::utils<IY>;
    static const int spline_order_x = static_cast<int>(IX);
    static const int spline_order_y = static_cast<int>(IY);

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  offset_t w, offset_t nw, offset_t sw, reduce_t wscl,
                  offset_t h, offset_t nh, offset_t sh, reduce_t hscl,
                  reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ixlow = static_cast<offset_t>(ceil(x - 0.5 * (spline_order_x + 1 ) * wscl));
        offset_t ixupp = static_cast<offset_t>(floor(x + 0.5 * (spline_order_x + 1 ) * wscl));
        reduce_t y = (h + shift) * hscl - shift;
        offset_t iylow = static_cast<offset_t>(ceil(y - 0.5 * (spline_order_y + 1 ) * hscl));
        offset_t iyupp = static_cast<offset_t>(floor(y + 0.5 * (spline_order_y + 1 ) * hscl));

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t iy = iylow; iy <= iyupp; ++iy) {
            reduce_t    dy = spline_utils_y::weight((y - iy) / hscl);
            signed char sy = bound_utils_y::sign(iy, nh);
            offset_t    oy = bound_utils_y::index(iy, nh) * sh;
            for (offset_t ix = ixlow; ix <= ixupp; ++ix) {
                reduce_t    dx = dy * spline_utils_x::weight((x - ix) / wscl);
                signed char sx = sy * bound_utils_x::sign(ix, nw);
                offset_t    ox = oy + bound_utils_x::index(ix, nw) * sw;
                acc += static_cast<reduce_t>(bound::get(inp, ox, sx)) * dx;
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

 /***                              ANY                              ***/
template <int U,
          spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ>
struct Multiscale<three, U, IX, BX, IY, BY, IZ, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils_x = spline::utils<IX>;
    using spline_utils_y = spline::utils<IY>;
    using spline_utils_z = spline::utils<IZ>;
    static const int spline_order_x = static_cast<int>(IX);
    static const int spline_order_y = static_cast<int>(IY);
    static const int spline_order_z = static_cast<int>(IZ);

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  offset_t w, offset_t nw, offset_t sw, reduce_t wscl,
                  offset_t h, offset_t nh, offset_t sh, reduce_t hscl,
                  offset_t d, offset_t nd, offset_t sd, reduce_t dscl,
                  reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ixlow = static_cast<offset_t>(ceil(x - 0.5 * (spline_order_x + 1) * wscl));
        offset_t ixupp = static_cast<offset_t>(floor(x + 0.5 * (spline_order_x + 1) * wscl));
        reduce_t y = (h + shift) * hscl - shift;
        offset_t iylow = static_cast<offset_t>(ceil(y - 0.5 * (spline_order_y + 1) * hscl));
        offset_t iyupp = static_cast<offset_t>(floor(y + 0.5 * (spline_order_y + 1) * hscl));
        reduce_t z = (d + shift) * dscl - shift;
        offset_t izlow = static_cast<offset_t>(ceil(z - 0.5 * (spline_order_z + 1) * dscl));
        offset_t izupp = static_cast<offset_t>(floor(z + 0.5 * (spline_order_z + 1) * dscl));

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t iz = izlow; iz <= izupp; ++iz) {
            reduce_t    dz = spline_utils_z::weight((z - iz) / dscl);
            signed char sz = bound_utils_z::sign(iz, nd);
            offset_t    oz = bound_utils_z::index(iz, nd) * sd;
            for (offset_t iy = iylow; iy <= iyupp; ++iy) {
                reduce_t    dy = dz * spline_utils_y::weight((y - iy) / hscl);
                signed char sy = sz * bound_utils_y::sign(iy, nh);
                offset_t    oy = oz + bound_utils_y::index(iy, nh) * sh;
                for (offset_t ix = ixlow; ix <= ixupp; ++ix) {
                    reduce_t    dx = dy * spline_utils_x::weight((x - ix) / wscl);
                    signed char sx = sy * bound_utils_x::sign(ix, nw);
                    offset_t    ox = oy + bound_utils_x::index(ix, nw) * sw;
                    acc += bound::cget<reduce_t>(inp, ox, sx) * dx;
                }
            }
        }
        *out = static_cast<scalar_t>(acc);
    }
};


/***                      LINEAR + BOUND 2                         ***/
template <bound::type BX, bound::type BY, bound::type BZ>
struct Multiscale<three, two, L, BX, L, BY, L, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils = spline::utils<L>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  offset_t w, offset_t nw, offset_t sw, reduce_t wscl,
                  offset_t h, offset_t nh, offset_t sh, reduce_t hscl,
                  offset_t d, offset_t nd, offset_t sd, reduce_t dscl,
                  reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x));
        offset_t ix0 =  ix1 - 1, ix2 = ix1 + 1, ix3 = ix1 + 2;
        reduce_t dx1 = spline_utils::weight((x - ix1) / wscl);
        reduce_t dx0 = spline_utils::weight((x - ix0) / wscl);
        reduce_t dx2 = spline_utils::weight((ix2 - x) / wscl);
        reduce_t dx3 = spline_utils::weight((ix3 - x) / wscl);
        signed char sx3 = bound_utils_x::sign(ix3, nw);
        signed char sx2 = bound_utils_x::sign(ix2, nw);
        signed char sx0 = bound_utils_x::sign(ix0, nw);
        signed char sx1 = bound_utils_x::sign(ix1, nw);
        ix3 = bound_utils_x::index(ix3, nw) * sw;
        ix2 = bound_utils_x::index(ix2, nw) * sw;
        ix0 = bound_utils_x::index(ix0, nw) * sw;
        ix1 = bound_utils_x::index(ix1, nw) * sw;

        reduce_t y = (h + shift) * hscl - shift;
        offset_t iy1 = static_cast<offset_t>(floor(y));
        offset_t iy0 =  iy1 - 1, iy2 = iy1 + 1, iy3 = iy1 + 2;
        reduce_t dy1 = spline_utils::weight((y - iy1) / hscl);
        reduce_t dy0 = spline_utils::weight((y - iy0) / hscl);
        reduce_t dy2 = spline_utils::weight((iy2 - y) / hscl);
        reduce_t dy3 = spline_utils::weight((iy3 - y) / hscl);
        signed char sy3 = bound_utils_y::sign(iy3, nh);
        signed char sy2 = bound_utils_y::sign(iy2, nh);
        signed char sy0 = bound_utils_y::sign(iy0, nh);
        signed char sy1 = bound_utils_y::sign(iy1, nh);
        iy3 = bound_utils_y::index(iy3, nh) * sh;
        iy2 = bound_utils_y::index(iy2, nh) * sh;
        iy0 = bound_utils_y::index(iy0, nh) * sh;
        iy1 = bound_utils_y::index(iy1, nh) * sh;

        reduce_t z = (d + shift) * dscl - shift;
        offset_t iz1 = static_cast<offset_t>(floor(z));
        offset_t iz0 =  iz1 - 1, iz2 = iz1 + 1, iz3 = iz1 + 2;
        reduce_t dz1 = spline_utils::weight((z - iz1) / dscl);
        reduce_t dz0 = spline_utils::weight((z - iz0) / dscl);
        reduce_t dz2 = spline_utils::weight((iz2 - z) / dscl);
        reduce_t dz3 = spline_utils::weight((iz3 - z) / dscl);
        signed char sz3 = bound_utils_z::sign(iz3, nd);
        signed char sz2 = bound_utils_z::sign(iz2, nd);
        signed char sz0 = bound_utils_z::sign(iz0, nd);
        signed char sz1 = bound_utils_z::sign(iz1, nd);
        iz3 = bound_utils_z::index(iz3, nd) * sd;
        iz2 = bound_utils_z::index(iz2, nd) * sd;
        iz0 = bound_utils_z::index(iz0, nd) * sd;
        iz1 = bound_utils_z::index(iz1, nd) * sd;

        auto accum1d = [&](offset_t i, signed char s)
        {
          return bound::cget<reduce_t>(inp, i + ix0, s * sx0) * dx0
               + bound::cget<reduce_t>(inp, i + ix1, s * sx1) * dx1
               + bound::cget<reduce_t>(inp, i + ix2, s * sx2) * dx2
               + bound::cget<reduce_t>(inp, i + ix3, s * sx3) * dx3;
        };

        auto accum2d = [&](offset_t i, signed char s)
        {
          return accum1d(iy0 + i, sy0 * s) * dy0
               + accum1d(iy1 + i, sy1 * s) * dy1
               + accum1d(iy2 + i, sy2 * s) * dy2
               + accum1d(iy3 + i, sy3 * s) * dy3;
        };

        *out = static_cast<scalar_t>(accum2d(iz0, sz0) * dz0
                                   + accum2d(iz1, sz1) * dz1
                                   + accum2d(iz2, sz2) * dz2
                                   + accum2d(iz3, sz3) * dz3);
    }
};

/***********************************************************************
 *
 *                                  ND
 *
 **********************************************************************/

 /***                              ANY                              ***/
template <int D, int U> struct Multiscale<D, U> {

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  const offset_t * coord, const offset_t * size, const offset_t * stride,
                  const spline::type * inter, const bound::type * bnd,
                  const reduce_t * scl,  reduce_t shift)
    {
        offset_t    offsets[D];
        signed char signs[D];
        reduce_t    weights[D];
        reduce_t acc = static_cast<reduce_t>(0);
        for (int d=0; d<D; ++d) {
            reduce_t x = (coord[d] + shift) * scl[d] - shift;
            int spline_order = static_cast<int>(inter[d]);
            offset_t ilow = static_cast<offset_t>(ceil(x - 0.5 * (spline_order + 1 ) * scl[d]));
            offset_t iupp = static_cast<offset_t>(floor(x + 0.5 * (spline_order + 1 ) * scl[d]));

            for (offset_t i = ilow; i <= iupp; ++i) {
                signs[d]   = (d > 0 ? signs[d-1]   : static_cast<signed char>(1))
                           * bound::sign(bnd[d], i, size[d]);
                weights[d] = (d > 0 ? weights[d-1] : static_cast<reduce_t>(1))
                           * spline::weight(inter[d], (x - i) / scl[d]);
                offsets[d] = (d > 0 ? offsets[d-1] : static_cast<offset_t>(0))
                           + bound::index(bnd[d], i, size[d]) * stride[d];
                if (d == D-1)
                    acc += static_cast<reduce_t>(bound::get(inp, offsets[D-1], signs[D-1])) * weights[D-1];
            }
        }
        *out = static_cast<offset_t>(acc);
    }
};

} // namespace restrict
} // namespace jf

#endif // JF_RESTRICT
