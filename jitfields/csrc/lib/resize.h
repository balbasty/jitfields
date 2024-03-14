#ifndef JF_RESIZE
#define JF_RESIZE
#include "cuda_switch.h"
#include "spline.h"
#include "bounds.h"
#include "utils.h"
#include "batch.h"

namespace jf {
namespace resize {

const spline::type Z = spline::type::Nearest;
const spline::type L = spline::type::Linear;
const spline::type Q = spline::type::Quadratic;
const spline::type C = spline::type::Cubic;
const bound::type B0 = bound::type::NoCheck;
const int one   = 1;
const int two   = 2;
const int three = 3;


/***********************************************************************
 *
 *                                  ND
 *
 **********************************************************************/

template <int D,
          spline::type IX=Z,  bound::type BX=B0,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
struct Multiscale
{
    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t coord[D], const offset_t size[D], const offset_t stride[D],
                const spline::type * inter, const bound::type * bnd,
                const reduce_t scl[D], reduce_t shift)
    {
        // Precompute weights and indices
        reduce_t    w[8*D];
        offset_t    i[8*D];
        signed char s[8*D];
        offset_t    db[D];
#       pragma unroll
        for (int d=0; d<D; ++d) {
            reduce_t    *wd = w + 8*d;
            offset_t    *id = i + 8*d;
            signed char *sd = s + 8*d;
            reduce_t x = scl[d] * (coord[d] + shift) - shift;
            offset_t b0, b1;
            spline::bounds(inter[d], x, b0, b1);
            db[d] = b1-b0+1;
            for (offset_t b = b0; b <= b1; ++b) {
                *(wd++) = spline::weight(inter[d], x - b);
                *(sd++) = bound::sign(bnd[d], b, size[d]);
                *(id++) = bound::index(bnd[d], b, size[d]);
            }
        }

        // Convolve coefficients with basis functions
        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t j=0; j<prod<D>(db); ++j)
        {
            offset_t sub[D]; index2sub<D>(j, db, sub);
            offset_t offset = static_cast<offset_t>(0);
            reduce_t weight = static_cast<reduce_t>(1);
            signed char sgn = static_cast<signed char>(1);
#           pragma unroll
            for (offset_t d=0; d<D; ++d)
            {
                offset_t k = sub[d];
                offset += i[8*d+k] * stride[d];
                weight *= w[8*d+k];
                sgn    *= s[8*d+k];
            }
            acc += bound::cget<reduce_t>(inp, offset, sgn) * weight;
        }
        *out = static_cast<scalar_t>(acc);
    }
};

/***********************************************************************
 *
 *                                  1D
 *
 **********************************************************************/

/***                            NEAREST                             ***/
template <bound::type B> struct Multiscale<one, Z, B> {
    using bound_utils = bound::utils<B>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[1], const offset_t size[1],
                const offset_t stride[1], const reduce_t scl[1],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        reduce_t x = (w + shift) * scl[0] - shift;
        offset_t ix = static_cast<offset_t>(floor(x+0.5));
        signed char sx = bound_utils::sign(ix, nw);
        ix = bound_utils::index(ix, nw) * sw;
        *out = bound::get(inp, ix, sx);
    }
};

/***                            LINEAR                              ***/
template <bound::type B> struct Multiscale<one, L, B> {
    using bound_utils = bound::utils<B>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[1], const offset_t size[1],
                const offset_t stride[1], const reduce_t scl[1],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        reduce_t x = (w + shift) * scl[0] - shift;
        offset_t ix0 = static_cast<offset_t>(floor(x));
        offset_t ix1 = ix0 + 1;
        reduce_t dx1 = x - ix0;
        reduce_t dx0 = 1 - dx1;
        signed char  sx0 = bound_utils::sign(ix0, nw);
        signed char  sx1 = bound_utils::sign(ix1, nw);
        ix0 = bound_utils::index(ix0, nw) * sw;
        ix1 = bound_utils::index(ix1, nw) * sw;

        *out = static_cast<scalar_t>(
                  static_cast<reduce_t>(bound::get(inp, ix0, sx0)) * dx0
                + static_cast<reduce_t>(bound::get(inp, ix1, sx1)) * dx1);
    }
};

/***                          QUADRATIC                             ***/
template <bound::type B> struct Multiscale<one, Q, B> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<Q>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[1], const offset_t size[1],
                const offset_t stride[1], const reduce_t scl[1],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        reduce_t x = (w + shift) * scl[0] - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x+0.5));
        reduce_t dx1 = spline_utils::weight(x - ix1);
        reduce_t dx0 = spline_utils::fastweight(x - (ix1 - 1));
        reduce_t dx2 = spline_utils::fastweight((ix1 + 1) - x);
        signed char  sx0 = bound_utils::sign(ix1-1, nw);
        signed char  sx2 = bound_utils::sign(ix1+1, nw);
        signed char  sx1 = bound_utils::sign(ix1,   nw);
        offset_t ix0, ix2;
        ix0 = bound_utils::index(ix1-1, nw) * sw;
        ix2 = bound_utils::index(ix1+1, nw) * sw;
        ix1 = bound_utils::index(ix1,   nw) * sw;

        *out = static_cast<scalar_t>(
                  static_cast<reduce_t>(bound::get(inp, ix0, sx0)) * dx0
                + static_cast<reduce_t>(bound::get(inp, ix1, sx1)) * dx1
                + static_cast<reduce_t>(bound::get(inp, ix2, sx2)) * dx2);
    }
};

/***                             CUBIC                              ***/
template <bound::type B> struct Multiscale<one, C, B> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<C>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[1], const offset_t size[1],
                const offset_t stride[1], const reduce_t scl[1],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        reduce_t x = (w + shift) * scl[0] - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x));
        reduce_t dx1 = spline_utils::fastweight(x - ix1);
        reduce_t dx0 = spline_utils::fastweight(x - (ix1 - 1));
        reduce_t dx2 = spline_utils::fastweight((ix1 + 1) - x);
        reduce_t dx3 = spline_utils::fastweight((ix1 + 2) - x);
        signed char  sx0 = bound_utils::sign(ix1-1, nw);
        signed char  sx2 = bound_utils::sign(ix1+1, nw);
        signed char  sx3 = bound_utils::sign(ix1+2, nw);
        signed char  sx1 = bound_utils::sign(ix1,   nw);
        offset_t ix0, ix2, ix3;
        ix0 = bound_utils::index(ix1-1, nw) * sw;
        ix2 = bound_utils::index(ix1+1, nw) * sw;
        ix3 = bound_utils::index(ix1+2, nw) * sw;
        ix1 = bound_utils::index(ix1,   nw) * sw;

        *out = static_cast<scalar_t>(
                  static_cast<reduce_t>(bound::get(inp, ix0, sx0)) * dx0
                + static_cast<reduce_t>(bound::get(inp, ix1, sx1)) * dx1
                + static_cast<reduce_t>(bound::get(inp, ix2, sx2)) * dx2
                + static_cast<reduce_t>(bound::get(inp, ix3, sx3)) * dx3);
    }
};

/***                             ANY                                ***/
template <spline::type IX, bound::type BX>
struct Multiscale<one, IX, BX> {
    using bound_utils_x = bound::utils<BX>;
    using spline_utils_x = spline::utils<IX>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[1], const offset_t size[1],
                const offset_t stride[1], const reduce_t scl[1],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];

        // Precompute weights and indices
        reduce_t x = scl[0] * (w + shift) - shift;
        offset_t bx0, bx1;
        spline_utils_x::bounds(x, bx0, bx1);
        offset_t dbx = bx1-bx0;
        reduce_t    wx[8];
        offset_t    ix[8];
        signed char sx[8];
        {
            reduce_t    *owx = wx;
            offset_t    *oix = ix;
            signed char *osx = sx;
            for (offset_t bx = bx0; bx <= bx1; ++bx) {
                reduce_t dx = fabs(x - bx);
                *(owx++)  = spline_utils_x::fastweight(dx);
                *(osx++)  = bound_utils_x::sign(bx, nw);
                *(oix++)  = bound_utils_x::index(bx, nw);
            }
        }

        // Convolve coefficients with basis functions
        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t i = 0; i <= dbx; ++i)
            acc += static_cast<reduce_t>(bound::get(inp, ix[i] * sw, sx[i])) * wx[i];
        *out = static_cast<scalar_t>(acc);
    }
};

/***********************************************************************
 *
 *                                  2D
 *
 **********************************************************************/

/***                           NEAREST                              ***/
template <bound::type BX, bound::type BY>
struct Multiscale<two, Z, BX, Z, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using spline_utils = spline::utils<Z>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[2], const offset_t size[2],
                const offset_t stride[2], const reduce_t scl[2],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];

        reduce_t x = (w + shift) * scl[0] - shift;
        reduce_t y = (h + shift) * scl[1] - shift;
        offset_t ix = static_cast<offset_t>(round(x));
        offset_t iy = static_cast<offset_t>(round(y));
        signed char  sx = bound_utils_x::sign(ix, nw);
        signed char  sy = bound_utils_y::sign(iy, nh);
        ix = bound_utils_x::index(ix, nw) * sw;
        iy = bound_utils_y::index(iy, nh) * sh;

        *out = bound::get(inp, ix + iy, sx * sy);
    }
};

/***                            LINEAR                              ***/
template <bound::type BX, bound::type BY>
struct Multiscale<two, L, BX, L, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using spline_utils = spline::utils<L>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[2], const offset_t size[2],
                const offset_t stride[2], const reduce_t scl[2],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];

        reduce_t x = (w + shift) * scl[0] - shift;
        reduce_t y = (h + shift) * scl[1] - shift;
        offset_t ix0 = static_cast<offset_t>(floor(x));
        offset_t iy0 = static_cast<offset_t>(floor(y));
        offset_t ix1 = ix0 + 1;
        offset_t iy1 = iy0 + 1;
        reduce_t dx1 = x - ix0;
        reduce_t dy1 = y - iy0;
        reduce_t dx0 = 1 - dx1;
        reduce_t dy0 = 1 - dy1;
        signed char  sx0 = bound_utils_x::sign(ix0, nw);
        signed char  sy0 = bound_utils_y::sign(iy0, nh);
        signed char  sx1 = bound_utils_x::sign(ix1, nw);
        signed char  sy1 = bound_utils_y::sign(iy1, nh);
        ix0 = bound_utils_x::index(ix0, nw) * sw;
        iy0 = bound_utils_y::index(iy0, nh) * sh;
        ix1 = bound_utils_x::index(ix1, nw) * sw;
        iy1 = bound_utils_y::index(iy1, nh) * sh;

        auto accum1d = [ix0, ix1, dx0, dx1, sx0, sx1, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * sx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * sx1)) * dx1;
        };

        *out = static_cast<scalar_t>(accum1d(iy0, sy0) * dy0
                                   + accum1d(iy1, sy1) * dy1);
    }
};

/***                          QUADRATIC                             ***/
template <bound::type BX, bound::type BY>
struct Multiscale<two, Q, BX, Q, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using spline_utils = spline::utils<Q>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[2], const offset_t size[2],
                const offset_t stride[2], const reduce_t scl[2],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];

        reduce_t x = (w + shift) * scl[0] - shift;
        reduce_t y = (h + shift) * scl[1] - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x+0.5));
        offset_t iy1 = static_cast<offset_t>(floor(y+0.5));
        reduce_t dx1 = spline_utils::weight(x - ix1);
        reduce_t dy1 = spline_utils::weight(y - iy1);
        reduce_t dx0 = spline_utils::fastweight(x - (ix1 - 1));
        reduce_t dy0 = spline_utils::fastweight(y - (iy1 - 1));
        reduce_t dx2 = spline_utils::fastweight((ix1 + 1) - x);
        reduce_t dy2 = spline_utils::fastweight((iy1 + 1) - y);
        signed char sx0 = bound_utils_x::sign(ix1-1, nw);
        signed char sy0 = bound_utils_y::sign(iy1-1, nh);
        signed char sx2 = bound_utils_x::sign(ix1+1, nw);
        signed char sy2 = bound_utils_y::sign(iy1+1, nh);
        signed char sx1 = bound_utils_x::sign(ix1,   nw);
        signed char sy1 = bound_utils_y::sign(iy1,   nh);
        offset_t ix0, iy0, ix2, iy2;
        ix0 = bound_utils_x::index(ix1-1, nw) * sw;
        iy0 = bound_utils_y::index(iy1-1, nh) * sh;
        ix2 = bound_utils_x::index(ix1+1, nw) * sw;
        iy2 = bound_utils_y::index(iy1+1, nh) * sh;
        ix1 = bound_utils_x::index(ix1,   nw) * sw;
        iy1 = bound_utils_y::index(iy1,   nh) * sh;

        auto accum1d = [ix0, ix1, ix2, dx0, dx1, dx2, sx0, sx1, sx2, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * sx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * sx1)) * dx1
               + static_cast<reduce_t>(bound::get(inp, i + ix2, s * sx2)) * dx2;
        };

        *out = static_cast<scalar_t>(accum1d(iy0, sy0) * dy0
                                   + accum1d(iy1, sy1) * dy1
                                   + accum1d(iy2, sy2) * dy2);
    }
};

/***                            CUBIC                               ***/
template <bound::type BX, bound::type BY>
struct Multiscale<two, C, BX, C, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using spline_utils = spline::utils<C>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[2], const offset_t size[2],
                const offset_t stride[2], const reduce_t scl[2],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];

        reduce_t x = (w + shift) * scl[0] - shift;
        reduce_t y = (h + shift) * scl[1] - shift;
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
        signed char  sx0 = bound_utils_x::sign(ix1-1, nw);
        signed char  sy0 = bound_utils_y::sign(iy1-1, nh);
        signed char  sx2 = bound_utils_x::sign(ix1+1, nw);
        signed char  sy2 = bound_utils_y::sign(iy1+1, nh);
        signed char  sx3 = bound_utils_x::sign(ix1+2, nw);
        signed char  sy3 = bound_utils_y::sign(iy1+2, nh);
        signed char  sx1 = bound_utils_x::sign(ix1,   nw);
        signed char  sy1 = bound_utils_y::sign(iy1,   nh);
        offset_t ix0, ix2, ix3, iy0, iy2, iy3;
        ix0 = bound_utils_x::index(ix1-1, nw) * sw;
        iy0 = bound_utils_y::index(iy1-1, nh) * sh;
        ix2 = bound_utils_x::index(ix1+1, nw) * sw;
        iy2 = bound_utils_y::index(iy1+1, nh) * sh;
        ix3 = bound_utils_x::index(ix1+2, nw) * sw;
        iy3 = bound_utils_y::index(iy1+2, nh) * sh;
        ix1 = bound_utils_x::index(ix1,   nw) * sw;
        iy1 = bound_utils_y::index(iy1,   nh) * sh;

        auto accum1d = [ix0, ix1, ix2, ix3, dx0, dx1, dx2, dx3,
                        sx0, sx1, sx2, sx3, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * sx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * sx1)) * dx1
               + static_cast<reduce_t>(bound::get(inp, i + ix2, s * sx2)) * dx2
               + static_cast<reduce_t>(bound::get(inp, i + ix3, s * sx3)) * dx3;
        };

        *out = static_cast<scalar_t>(accum1d(iy0, sy0) * dy0
                                   + accum1d(iy1, sy1) * dy1
                                   + accum1d(iy2, sy2) * dy2
                                   + accum1d(iy3, sy3) * dy3);
    }
};

/***                             ANY                                ***/
template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY>
struct Multiscale<two, IX, BX, IY, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using spline_utils_x = spline::utils<IX>;
    using spline_utils_y = spline::utils<IY>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[2],    const offset_t size[2],
                const offset_t stride[2], const reduce_t scl[2],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];

        // Precompute weights and indices
        reduce_t x = scl[0] * (w + shift) - shift;
        reduce_t y = scl[1] * (h + shift) - shift;
        offset_t bx0, bx1, by0, by1;
        spline_utils_x::bounds(x, bx0, bx1);
        spline_utils_y::bounds(y, by0, by1);
        offset_t dbx = bx1-bx0;
        offset_t dby = by1-by0;
        reduce_t    wx[8],  wy[8];
        offset_t    ix[8],  iy[8];
        signed char sx[8],  sy[8];
        {
            reduce_t    *owy = wy;
            offset_t    *oiy = iy;
            signed char *osy = sy;
            for (offset_t by = by0; by <= by1; ++by) {
                scalar_t dy = fabs(y - by);
                *(owy++)  = spline_utils_y::fastweight(dy);
                *(osy++)  = bound_utils_y::sign(by, nh);
                *(oiy++)  = bound_utils_y::index(by, nh);
            }
        }
        {
            reduce_t    *owx = wx;
            offset_t    *oix = ix;
            signed char *osx = sx;
            for (offset_t bx = bx0; bx <= bx1; ++bx) {
                scalar_t dx = fabs(x - bx);
                *(owx++)  = spline_utils_x::fastweight(dx);
                *(osx++)  = bound_utils_x::sign(bx, nw);
                *(oix++)  = bound_utils_x::index(bx, nw);
            }
        }

        // Convolve coefficients with basis functions
        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t j = 0; j <= dby; ++j) {
            offset_t    oyy = iy[j] * sh;
            signed char syy = sy[j];
            reduce_t    wyy = wy[j];
            for (offset_t i = 0; i <= dbx; ++i) {
                offset_t    oxy = oyy + ix[i] * sw;
                signed char sxy = syy * sx[i];
                reduce_t    wxy = wyy * wx[i];
                acc += static_cast<reduce_t>(bound::get(inp, oxy, sxy)) * wxy;
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
struct Multiscale<two, Z, BX, Z, BY, Z, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils = spline::utils<Z>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[3], const offset_t size[3],
                const offset_t stride[3], const reduce_t scl[3],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];
        offset_t d = loc[2], nd = size[2], sd = stride[2];

        reduce_t x = (w + shift) * scl[0] - shift;
        reduce_t y = (h + shift) * scl[1] - shift;
        reduce_t z = (d + shift) * scl[2] - shift;
        offset_t ix = static_cast<offset_t>(round(x));
        offset_t iy = static_cast<offset_t>(round(y));
        offset_t iz = static_cast<offset_t>(round(z));
        signed char  sx = bound_utils_x::sign(ix, nw);
        signed char  sy = bound_utils_y::sign(iy, nh);
        signed char  sz = bound_utils_z::sign(iz, nd);
        ix = bound_utils_x::index(ix, nw) * sw;
        iy = bound_utils_y::index(iy, nh) * sh;
        iz = bound_utils_z::index(iz, nd) * sd;

        *out = bound::get(inp, ix + iy + iz, sx * sy * sz);
    }
};

/***                            LINEAR                              ***/
template <bound::type BX, bound::type BY, bound::type BZ>
struct Multiscale<three, L, BX, L, BY, L, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils = spline::utils<L>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[3], const offset_t size[3],
                const offset_t stride[3], const reduce_t scl[3],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];
        offset_t d = loc[2], nd = size[2], sd = stride[2];

        reduce_t x = (w + shift) * scl[0] - shift;
        reduce_t y = (h + shift) * scl[1] - shift;
        reduce_t z = (d + shift) * scl[2] - shift;
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
        signed char  sx0 = bound_utils_x::sign(ix0, nw);
        signed char  sy0 = bound_utils_y::sign(iy0, nh);
        signed char  sz0 = bound_utils_z::sign(iz0, nd);
        signed char  sx1 = bound_utils_x::sign(ix1, nw);
        signed char  sy1 = bound_utils_y::sign(iy1, nh);
        signed char  sz1 = bound_utils_z::sign(iz1, nd);
        ix0 = bound_utils_x::index(ix0, nw) * sw;
        iy0 = bound_utils_y::index(iy0, nh) * sh;
        iz0 = bound_utils_z::index(iz0, nd) * sd;
        ix1 = bound_utils_x::index(ix1, nw) * sw;
        iy1 = bound_utils_y::index(iy1, nh) * sh;
        iz1 = bound_utils_z::index(iz1, nd) * sd;

        auto accum1d = [ix0, ix1, dx0, dx1, sx0, sx1, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * sx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * sx1)) * dx1;
        };

        auto accum2d = [iy0, iy1, dy0, dy1, sy0, sy1, accum1d]
                        (offset_t i, signed char s)
        {
          return accum1d(iy0 + i, sy0 * s) * dy0
               + accum1d(iy1 + i, sy1 * s) * dy1;
        };

        *out  = static_cast<scalar_t>(accum2d(iz0, sz0) * dz0
                                    + accum2d(iz1, sz1) * dz1);
    }
};

/***                          QUADRATIC                             ***/
template <bound::type BX, bound::type BY, bound::type BZ>
struct Multiscale<three, Q, BX, Q, BY, Q, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils = spline::utils<Q>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[3], const offset_t size[3],
                const offset_t stride[3], const reduce_t scl[3],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];
        offset_t d = loc[2], nd = size[2], sd = stride[2];

        reduce_t x = (w + shift) * scl[0] - shift;
        reduce_t y = (h + shift) * scl[1] - shift;
        reduce_t z = (d + shift) * scl[2] - shift;
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
        signed char  sx0 = bound_utils_x::sign(ix1-1, nw);
        signed char  sy0 = bound_utils_y::sign(iy1-1, nh);
        signed char  sz0 = bound_utils_z::sign(iz1-1, nd);
        signed char  sx2 = bound_utils_x::sign(ix1+1, nw);
        signed char  sy2 = bound_utils_y::sign(iy1+1, nh);
        signed char  sz2 = bound_utils_z::sign(iz1+1, nd);
        signed char  sx1 = bound_utils_x::sign(ix1,   nw);
        signed char  sy1 = bound_utils_y::sign(iy1,   nh);
        signed char  sz1 = bound_utils_z::sign(iz1,   nd);
        offset_t ix0, iy0, iz0, ix2, iy2, iz2;
        ix0 = bound_utils_x::index(ix1-1, nw) * sw;
        iy0 = bound_utils_y::index(iy1-1, nh) * sh;
        iz0 = bound_utils_z::index(iz1-1, nd) * sd;
        ix2 = bound_utils_x::index(ix1+1, nw) * sw;
        iy2 = bound_utils_y::index(iy1+1, nh) * sh;
        iz2 = bound_utils_z::index(iz1+1, nd) * sd;
        ix1 = bound_utils_x::index(ix1,   nw) * sw;
        iy1 = bound_utils_y::index(iy1,   nh) * sh;
        iz1 = bound_utils_z::index(iz1,   nd) * sd;

        auto accum1d = [ix0, ix1, ix2, dx0, dx1, dx2, sx0, sx1, sx2, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * sx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * sx1)) * dx1
               + static_cast<reduce_t>(bound::get(inp, i + ix2, s * sx2)) * dx2;
        };

        auto accum2d = [iy0, iy1, iy2, dy0, dy1, dy2, sy0, sy1, sy2, accum1d]
                        (offset_t i, signed char s)
        {
          return accum1d(iy0 + i, sy0 * s) * dy0
               + accum1d(iy1 + i, sy1 * s) * dy1
               + accum1d(iy2 + i, sy2 * s) * dy2;
        };

        *out  = static_cast<scalar_t>(accum2d(iz0, sz0) * dz0
                                    + accum2d(iz1, sz1) * dz1
                                    + accum2d(iz2, sz2) * dz2);
    }
};

/***                            CUBIC                               ***/
template <bound::type BX, bound::type BY, bound::type BZ>
struct Multiscale<three, C, BX, C, BY, C, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils = spline::utils<C>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[3], const offset_t size[3],
                const offset_t stride[3], const reduce_t scl[3],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];
        offset_t d = loc[2], nd = size[2], sd = stride[2];

        reduce_t x = (w + shift) * scl[0] - shift;
        reduce_t y = (h + shift) * scl[1] - shift;
        reduce_t z = (d + shift) * scl[2] - shift;
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
        signed char  sx0 = bound_utils_x::sign(ix1-1, nw);
        signed char  sy0 = bound_utils_y::sign(iy1-1, nh);
        signed char  sz0 = bound_utils_z::sign(iz1-1, nd);
        signed char  sx2 = bound_utils_x::sign(ix1+1, nw);
        signed char  sy2 = bound_utils_y::sign(iy1+1, nh);
        signed char  sz2 = bound_utils_z::sign(iz1+1, nd);
        signed char  sx3 = bound_utils_x::sign(ix1+2, nw);
        signed char  sy3 = bound_utils_y::sign(iy1+2, nh);
        signed char  sz3 = bound_utils_z::sign(iz1+2, nd);
        signed char  sx1 = bound_utils_x::sign(ix1,   nw);
        signed char  sy1 = bound_utils_y::sign(iy1,   nh);
        signed char  sz1 = bound_utils_z::sign(iz1,   nd);
        offset_t ix0, ix2, ix3, iy0, iy2, iy3, iz0, iz2, iz3;
        ix0 = bound_utils_x::index(ix1-1, nw) * sw;
        iy0 = bound_utils_y::index(iy1-1, nh) * sh;
        iz0 = bound_utils_z::index(iz1-1, nd) * sd;
        ix2 = bound_utils_x::index(ix1+1, nw) * sw;
        iy2 = bound_utils_y::index(iy1+1, nh) * sh;
        iz2 = bound_utils_z::index(iz1+1, nd) * sd;
        ix3 = bound_utils_x::index(ix1+2, nw) * sw;
        iy3 = bound_utils_y::index(iy1+2, nh) * sh;
        iz3 = bound_utils_z::index(iz1+2, nd) * sd;
        ix1 = bound_utils_x::index(ix1,   nw) * sw;
        iy1 = bound_utils_y::index(iy1,   nh) * sh;
        iz1 = bound_utils_z::index(iz1,   nd) * sd;

        auto accum1d = [ix0, ix1, ix2, ix3, dx0, dx1, dx2, dx3,
                        sx0, sx1, sx2, sx3, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * sx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * sx1)) * dx1
               + static_cast<reduce_t>(bound::get(inp, i + ix2, s * sx2)) * dx2
               + static_cast<reduce_t>(bound::get(inp, i + ix3, s * sx3)) * dx3;
        };

        auto accum2d = [iy0, iy1, iy2, iy3, dy0, dy1, dy2, dy3,
                        sy0, sy1, sy2, sy3, accum1d]
                        (offset_t i, signed char s)
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

/***                             ANY                                ***/
template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ>
struct Multiscale<three, IX, BX, IY, BY, IZ, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using spline_utils_x = spline::utils<IX>;
    using spline_utils_y = spline::utils<IY>;
    using spline_utils_z = spline::utils<IZ>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static inline __device__
    void resize(scalar_t * out, const scalar_t * inp,
                const offset_t loc[3], const offset_t size[3],
                const offset_t stride[3], const reduce_t scl[3],
                reduce_t shift)
    {
        offset_t w = loc[0], nw = size[0], sw = stride[0];
        offset_t h = loc[1], nh = size[1], sh = stride[1];
        offset_t d = loc[2], nd = size[2], sd = stride[2];

        // Precompute weights and indices
        reduce_t x = scl[0] * (w + shift) - shift;
        reduce_t y = scl[1] * (h + shift) - shift;
        reduce_t z = scl[2] * (d + shift) - shift;
        offset_t bx0, bx1, by0, by1, bz0, bz1;
        spline_utils_x::bounds(x, bx0, bx1);
        spline_utils_y::bounds(y, by0, by1);
        spline_utils_z::bounds(z, bz0, bz1);
        offset_t dbx = bx1-bx0;
        offset_t dby = by1-by0;
        offset_t dbz = bz1-bz0;
        reduce_t    wx[8],  wy[8],  wz[8];
        offset_t    ix[8],  iy[8],  iz[8];
        signed char sx[8],  sy[8],  sz[8];
        {
            reduce_t    *owz = wz;
            offset_t    *oiz = iz;
            signed char *osz = sz;
            for (offset_t bz = bz0; bz <= bz1; ++bz) {
                scalar_t dz = fabs(z - bz);
                *(owz++)  = spline_utils_z::fastweight(dz);
                *(osz++)  = bound_utils_z::sign(bz, nd);
                *(oiz++)  = bound_utils_z::index(bz, nd);
            }
        }
        {
            reduce_t    *owy = wy;
            offset_t    *oiy = iy;
            signed char *osy = sy;
            for (offset_t by = by0; by <= by1; ++by) {
                scalar_t dy = fabs(y - by);
                *(owy++)  = spline_utils_y::fastweight(dy);
                *(osy++)  = bound_utils_y::sign(by, nh);
                *(oiy++)  = bound_utils_y::index(by, nh);
            }
        }
        {
            reduce_t    *owx = wx;
            offset_t    *oix = ix;
            signed char *osx = sx;
            for (offset_t bx = bx0; bx <= bx1; ++bx) {
                scalar_t dx = fabs(x - bx);
                *(owx++)  = spline_utils_x::fastweight(dx);
                *(osx++)  = bound_utils_x::sign(bx, nw);
                *(oix++)  = bound_utils_x::index(bx, nw);
            }
        }

        // Convolve coefficients with basis functions
        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t k = 0; k <= dbz; ++k) {
            offset_t    ozz = iz[k] * sd;
            signed char szz = sz[k];
            reduce_t    wzz = wz[k];
            for (offset_t j = 0; j <= dby; ++j) {
                offset_t    oyz = ozz + iy[j] * sh;
                signed char syz = szz * sy[j];
                reduce_t    wyz = wzz * wy[j];
                for (offset_t i = 0; i <= dbx; ++i) {
                    offset_t    oxyz = oyz + ix[i] * sw;
                    signed char sxyz = syz * sx[i];
                    reduce_t    wxyz = wyz * wx[i];
                    acc += static_cast<reduce_t>(bound::get(inp, oxyz, sxyz)) * wxyz;
                }
            }
        }
        *out = static_cast<scalar_t>(acc);
    }
};

} // namespace resize
} // namespace jf

#endif // JF_RESIZE
