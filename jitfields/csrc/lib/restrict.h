#ifndef JF_RESTRICT
#define JF_RESTRICT
#include "cuda_switch.h"
#include "spline.h"
#include "bounds.h"

namespace jf {
namespace restrict {

const spline::type Z = spline::type::Nearest;
const spline::type L = spline::type::Linear;
const int zero  = 0;
const int one   = 1;
const int two   = 2;
const int three = 3;

/***********************************************************************
 *
 *                                  ND
 *
 **********************************************************************/
 /***                              ANY                              ***/
// D - Number of spatial dimensions
// U - Upper bound on the restriction factor
// IX, IY, IZ - Interpolation order
template <int D, int U=zero,
          spline::type IX=Z, spline::type IY=IX, spline::type IZ=IY>
struct Multiscale
{
    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  const offset_t * coord, const offset_t * size, const offset_t * stride,
                  const spline::type * inter,
                  const reduce_t * scl,  reduce_t shift, signed char sgn = 1)
    {
        offset_t numel = 1;
        offset_t ilow[D], iupp[D], isize[D];
        reduce_t x[D];
        for (offset_t d=0; d<D; ++d)
        {
            int spline_order = static_cast<int>(inter[d]);
            x[d] = (coord[d] + shift) * scl[d] - shift;
            ilow[d] = max(
                static_cast<offset_t>(0),
                static_cast<offset_t>(ceil(x[d] - 0.5 * (spline_order + 1) * scl[d])));
            iupp[d] = min(
                static_cast<offset_t>(size[d]-1),
                static_cast<offset_t>(floor(x[d] + 0.5 * (spline_order + 1) * scl[d])));
            isize[d] = 1 + iupp[d] - ilow[d];
            numel *= isize[d];
        }

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t j=0; j<numel; ++j)
        {
            offset_t sub[D]; index2sub<D>(j, isize, sub);
            offset_t offset = static_cast<offset_t>(0);
            reduce_t weight = static_cast<reduce_t>(1);
            for (offset_t d=0; d<D; ++d)
            {
                offset_t i = ilow[d] + sub[d];
                weight *= spline::weight(inter[d], (x[d] - i) / scl[d]);
                offset += i * stride[d];
            }
            acc += static_cast<reduce_t>(inp[offset]) * weight;
        }
        bound::add(out, static_cast<offset_t>(0), acc, sgn);
    }
};

/***********************************************************************
 *
 *                                  1D
 *
 **********************************************************************/

 /***                              ANY                              ***/
template <int U, spline::type I>
struct Multiscale<one, U, I>
{
    using spline_utils = spline::utils<I>;
    static const int spline_order = static_cast<int>(I);

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  const offset_t loc[1], const offset_t size[1],
                  const offset_t stride[1], const reduce_t scl[1],
                  reduce_t shift, signed char sgn = 1)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        reduce_t wscl = scl[0];
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ixlow = max(
            static_cast<offset_t>(0),
            static_cast<offset_t>(ceil(x - 0.5 * (spline_order + 1 ) * wscl)));
        offset_t ixupp = min(
            static_cast<offset_t>(nw-1),
            static_cast<offset_t>(floor(x + 0.5 * (spline_order + 1 ) * wscl)));

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t ix = ixlow; ix <= ixupp; ++ix)
        {
            reduce_t dx = spline_utils::weight((x - ix) / wscl);
            acc += static_cast<reduce_t>(inp[ix * sw]) * dx;
        }
        bound::add(out, static_cast<offset_t>(0), acc, sgn);
    }
};


/***                      LINEAR + BOUND 2                         ***/
template <>
struct Multiscale<one, two, L> {
    using spline_utils = spline::utils<L>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  const offset_t loc[1], const offset_t size[1],
                  const offset_t stride[1], const reduce_t scl[1],
                  reduce_t shift, signed char sgn = 1)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        reduce_t wscl = scl[0];

        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x));
        offset_t ix0 =  ix1 - 1, ix2 = ix1 + 1, ix3 = ix1 + 2;
        reduce_t dx1 = spline_utils::weight((x - ix1) / wscl);
        reduce_t dx0 = spline_utils::weight((x - ix0) / wscl);
        reduce_t dx2 = spline_utils::weight((ix2 - x) / wscl);
        reduce_t dx3 = spline_utils::weight((ix3 - x) / wscl);

        auto accum1d = [&]()
        {
            reduce_t acc = static_cast<reduce_t>(0);
            if (0 <= ix0 && ix0 < nw) acc += static_cast<reduce_t>(inp[ix0*sw]) * dx0;
            if (0 <= ix1 && ix1 < nw) acc += static_cast<reduce_t>(inp[ix1*sw]) * dx1;
            if (0 <= ix2 && ix2 < nw) acc += static_cast<reduce_t>(inp[ix2*sw]) * dx2;
            if (0 <= ix3 && ix3 < nw) acc += static_cast<reduce_t>(inp[ix3*sw]) * dx3;
            return acc;
        };

        bound::add(out, static_cast<offset_t>(0), accum1d(), sgn);
    }
};

/***********************************************************************
 *
 *                                  2D
 *
 **********************************************************************/

/***                              ANY                              ***/
template <int U, spline::type IX, spline::type IY>
struct Multiscale<two, U, IX, IY> {
    using spline_utils_x = spline::utils<IX>;
    using spline_utils_y = spline::utils<IY>;
    static const int spline_order_x = static_cast<int>(IX);
    static const int spline_order_y = static_cast<int>(IY);

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  const offset_t loc[2], const offset_t size[2],
                  const offset_t stride[2], const reduce_t scl[2],
                  reduce_t shift, signed char sgn = 1)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];
        reduce_t wscl = scl[0], hscl = scl[1];
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ixlow = max(
            static_cast<offset_t>(0),
            static_cast<offset_t>(ceil(x - 0.5 * (spline_order_x + 1) * wscl)));
        offset_t ixupp = min(
            static_cast<offset_t>(nw-1),
            static_cast<offset_t>(floor(x + 0.5 * (spline_order_x + 1) * wscl)));
        reduce_t y = (h + shift) * hscl - shift;
        offset_t iylow = max(
            static_cast<offset_t>(0),
            static_cast<offset_t>(ceil(y - 0.5 * (spline_order_y + 1) * hscl)));
        offset_t iyupp = min(
            static_cast<offset_t>(nh-1),
            static_cast<offset_t>(floor(y + 0.5 * (spline_order_y + 1) * hscl)));

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t iy = iylow; iy <= iyupp; ++iy)
        {
            reduce_t    dy = spline_utils_y::weight((y - iy) / hscl);
            offset_t    oy = iy * sh;
            for (offset_t ix = ixlow; ix <= ixupp; ++ix)
            {
                reduce_t    dx = dy * spline_utils_x::weight((x - ix) / wscl);
                offset_t    ox = oy + ix * sw;
                acc += static_cast<reduce_t>(inp[ox]) * dx;
            }
        }
        bound::add(out, static_cast<offset_t>(0), acc, sgn);
    }
};


/***                      LINEAR + BOUND 2                         ***/
template <>
struct Multiscale<two, two, L> {
    using spline_utils = spline::utils<L>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  const offset_t loc[2], const offset_t size[2],
                  const offset_t stride[2], const reduce_t scl[2],
                  reduce_t shift, signed char sgn = 1)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];
        reduce_t wscl = scl[0], hscl = scl[1];

        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x));
        offset_t ix0 =  ix1 - 1, ix2 = ix1 + 1, ix3 = ix1 + 2;
        reduce_t dx1 = spline_utils::weight((x - ix1) / wscl);
        reduce_t dx0 = spline_utils::weight((x - ix0) / wscl);
        reduce_t dx2 = spline_utils::weight((ix2 - x) / wscl);
        reduce_t dx3 = spline_utils::weight((ix3 - x) / wscl);

        reduce_t y = (h + shift) * hscl - shift;
        offset_t iy1 = static_cast<offset_t>(floor(y));
        offset_t iy0 =  iy1 - 1, iy2 = iy1 + 1, iy3 = iy1 + 2;
        reduce_t dy1 = spline_utils::weight((y - iy1) / hscl);
        reduce_t dy0 = spline_utils::weight((y - iy0) / hscl);
        reduce_t dy2 = spline_utils::weight((iy2 - y) / hscl);
        reduce_t dy3 = spline_utils::weight((iy3 - y) / hscl);

        auto accum1d = [&](offset_t i)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            if (0 <= ix0 && ix0 < nw) acc += static_cast<reduce_t>(inp[i+ix0*sw]) * dx0;
            if (0 <= ix1 && ix1 < nw) acc += static_cast<reduce_t>(inp[i+ix1*sw]) * dx1;
            if (0 <= ix2 && ix2 < nw) acc += static_cast<reduce_t>(inp[i+ix2*sw]) * dx2;
            if (0 <= ix3 && ix3 < nw) acc += static_cast<reduce_t>(inp[i+ix3*sw]) * dx3;
            return acc;
        };

        auto accum2d = [&]()
        {
            reduce_t acc = static_cast<reduce_t>(0);
            if (0 <= iy0 && iy0 < nh) acc += accum1d(iy0*sh) * dy0;
            if (0 <= iy1 && iy1 < nh) acc += accum1d(iy1*sh) * dy1;
            if (0 <= iy2 && iy2 < nh) acc += accum1d(iy2*sh) * dy2;
            if (0 <= iy3 && iy3 < nh) acc += accum1d(iy3*sh) * dy3;
            return acc;
        };

        bound::add(out, static_cast<offset_t>(0), accum2d(), sgn);
    }
};

/***********************************************************************
 *
 *                                  3D
 *
 **********************************************************************/

 /***                              ANY                              ***/
template <int U, spline::type IX, spline::type IY, spline::type IZ>
struct Multiscale<three, U, IX, IY, IZ> {
    using spline_utils_x = spline::utils<IX>;
    using spline_utils_y = spline::utils<IY>;
    using spline_utils_z = spline::utils<IZ>;
    static const int spline_order_x = static_cast<int>(IX);
    static const int spline_order_y = static_cast<int>(IY);
    static const int spline_order_z = static_cast<int>(IZ);

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  const offset_t loc[3], const offset_t size[3],
                  const offset_t stride[3], const reduce_t scl[3],
                  reduce_t shift, signed char sgn = 1)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];
        offset_t d = loc[2], nd = size[2], sd = stride[2];
        reduce_t wscl = scl[0], hscl = scl[1], dscl = scl[2];
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ixlow = max(
            static_cast<offset_t>(0),
            static_cast<offset_t>(ceil(x - 0.5 * (spline_order_x + 1) * wscl)));
        offset_t ixupp = min(
            static_cast<offset_t>(nw - 1),
            static_cast<offset_t>(floor(x + 0.5 * (spline_order_x + 1) * wscl)));
        reduce_t y = (h + shift) * hscl - shift;
        offset_t iylow = max(
            static_cast<offset_t>(0),
            static_cast<offset_t>(ceil(y - 0.5 * (spline_order_y + 1) * hscl)));
        offset_t iyupp = min(
            static_cast<offset_t>(nh - 1),
            static_cast<offset_t>(floor(y + 0.5 * (spline_order_y + 1) * hscl)));
        reduce_t z = (d + shift) * dscl - shift;
        offset_t izlow = max(
            static_cast<offset_t>(0),
            static_cast<offset_t>(ceil(z - 0.5 * (spline_order_z + 1) * dscl)));
        offset_t izupp = min(
            static_cast<offset_t>(nd - 1),
            static_cast<offset_t>(floor(z + 0.5 * (spline_order_z + 1) * dscl)));

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t iz = izlow; iz <= izupp; ++iz) {
            reduce_t    dz = spline_utils_z::weight((z - iz) / dscl);
            offset_t    oz = iz * sd;
            for (offset_t iy = iylow; iy <= iyupp; ++iy) {
                reduce_t    dy = dz * spline_utils_y::weight((y - iy) / hscl);
                offset_t    oy = oz + iy * sh;
                for (offset_t ix = ixlow; ix <= ixupp; ++ix) {
                    reduce_t    dx = dy * spline_utils_x::weight((x - ix) / wscl);
                    offset_t    ox = oy + ix * sw;
                    acc += static_cast<reduce_t>(inp[ox]) * dx;
                }
            }
        }
        bound::add(out, static_cast<offset_t>(0), acc, sgn);
    }
};


/***                      LINEAR + BOUND 2                         ***/
template <>
struct Multiscale<three, two, L> {
    using spline_utils = spline::utils<L>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void restrict(scalar_t * out, const scalar_t * inp,
                  const offset_t loc[3], const offset_t size[3],
                  const offset_t stride[3], const reduce_t scl[3],
                  reduce_t shift, signed char sgn = 1)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];
        offset_t d = loc[2], nd = size[2], sd = stride[2];
        reduce_t wscl = scl[0], hscl = scl[1], dscl = scl[2];

        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x));
        offset_t ix0 =  ix1 - 1, ix2 = ix1 + 1, ix3 = ix1 + 2;
        reduce_t dx1 = spline_utils::weight((x - ix1) / wscl);
        reduce_t dx0 = spline_utils::weight((x - ix0) / wscl);
        reduce_t dx2 = spline_utils::weight((ix2 - x) / wscl);
        reduce_t dx3 = spline_utils::weight((ix3 - x) / wscl);

        reduce_t y = (h + shift) * hscl - shift;
        offset_t iy1 = static_cast<offset_t>(floor(y));
        offset_t iy0 =  iy1 - 1, iy2 = iy1 + 1, iy3 = iy1 + 2;
        reduce_t dy1 = spline_utils::weight((y - iy1) / hscl);
        reduce_t dy0 = spline_utils::weight((y - iy0) / hscl);
        reduce_t dy2 = spline_utils::weight((iy2 - y) / hscl);
        reduce_t dy3 = spline_utils::weight((iy3 - y) / hscl);

        reduce_t z = (d + shift) * dscl - shift;
        offset_t iz1 = static_cast<offset_t>(floor(z));
        offset_t iz0 =  iz1 - 1, iz2 = iz1 + 1, iz3 = iz1 + 2;
        reduce_t dz1 = spline_utils::weight((z - iz1) / dscl);
        reduce_t dz0 = spline_utils::weight((z - iz0) / dscl);
        reduce_t dz2 = spline_utils::weight((iz2 - z) / dscl);
        reduce_t dz3 = spline_utils::weight((iz3 - z) / dscl);

        auto accum1d = [&](offset_t i)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            if (0 <= ix0 && ix0 < nw) acc += static_cast<reduce_t>(inp[i+ix0*sw]) * dx0;
            if (0 <= ix1 && ix1 < nw) acc += static_cast<reduce_t>(inp[i+ix1*sw]) * dx1;
            if (0 <= ix2 && ix2 < nw) acc += static_cast<reduce_t>(inp[i+ix2*sw]) * dx2;
            if (0 <= ix3 && ix3 < nw) acc += static_cast<reduce_t>(inp[i+ix3*sw]) * dx3;
            return acc;
        };

        auto accum2d = [&](offset_t i)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            if (0 <= iy0 && iy0 < nh) acc += accum1d(iy0*sh + i) * dy0;
            if (0 <= iy1 && iy1 < nh) acc += accum1d(iy1*sh + i) * dy1;
            if (0 <= iy2 && iy2 < nh) acc += accum1d(iy2*sh + i) * dy2;
            if (0 <= iy3 && iy3 < nh) acc += accum1d(iy3*sh + i) * dy3;
            return acc;
        };

        auto accum3d = [&]() {
            reduce_t acc = static_cast<reduce_t>(0);
            if (0 <= iz0 && iz0 < nd) acc += accum2d(iz0*sd) * dz0;
            if (0 <= iz1 && iz1 < nd) acc += accum2d(iz1*sd) * dz1;
            if (0 <= iz2 && iz2 < nd) acc += accum2d(iz2*sd) * dz2;
            if (0 <= iz3 && iz3 < nd) acc += accum2d(iz3*sd) * dz3;
            return acc;
        };

        bound::add(out, static_cast<offset_t>(0), accum3d(), sgn);
    }
};

} // namespace restrict
} // namespace jf

#endif // JF_RESTRICT
