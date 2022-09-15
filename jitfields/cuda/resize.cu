/* DEPENDENCIES:
 * #include "interpolation.cuh"
 * #include "bounds.cuh"
 * #include "batch.cuh"
 */

/* TODO
 * - try to use fma (fused multiply-add) throughout
 * - check if using an inner loop across batch elements is more efficient
 *   (we currently use an outer loop, so we recompute indices many times)
 */

const interpolation::type Z = interpolation::type::Nearest;
const interpolation::type L = interpolation::type::Linear;
const interpolation::type Q = interpolation::type::Quadratic;
const interpolation::type C = interpolation::type::Cubic;
const bound::type B0 = bound::type::NoCheck;
const int one = 1;
const int two = 2;
const int three = 3;

template <int D,
          interpolation::type IX=Z,  bound::type BX=B0,
          interpolation::type IY=IX, bound::type BY=BX,
          interpolation::type IZ=IY, bound::type BZ=BY>
struct Multiscale {};

/***********************************************************************
 *
 *                                  1D
 *
 **********************************************************************/

/***                            NEAREST                             ***/
template <bound::type B> struct Multiscale<one, Z, B, Z, B, Z, B> {
    using bound_utils = bound::utils<B>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, // loc/size/stride
                reduce_t wscl, reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix = static_cast<offset_t>(floor(x+0.5));
        signed char sx = bound_utils::sign( ix, nw);
        ix = bound_utils::index(ix, nw) * sw;
        *out = bound::get(inp, ix, sx);
    }
};

/***                            LINEAR                              ***/
template <bound::type B> struct Multiscale<one, L, B, L, B, L, B> {
    using bound_utils = bound::utils<B>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, // loc/size/stride
                reduce_t wscl, reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
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
template <bound::type B> struct Multiscale<one, Q, B, Q, B, Q, B> {
    using bound_utils = bound::utils<B>;
    using inter_utils = interpolation::utils<Q>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, // loc/size/stride
                reduce_t wscl, reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x+0.5));
        reduce_t dx1 = inter_utils::weight(x - ix1);
        reduce_t dx0 = inter_utils::fastweight(x - (ix1 - 1));
        reduce_t dx2 = inter_utils::fastweight((ix1 + 1) - x);
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
template <bound::type B> struct Multiscale<one, C, B, C, B, C, B> {
    using bound_utils = bound::utils<B>;
    using inter_utils = interpolation::utils<C>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, // loc/size/stride
                reduce_t wscl, reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x));
        reduce_t dx1 = inter_utils::fastweight(x - ix1);
        reduce_t dx0 = inter_utils::fastweight(x - (ix1 - 1));
        reduce_t dx2 = inter_utils::fastweight((ix1 + 1) - x);
        reduce_t dx3 = inter_utils::fastweight((ix1 + 2) - x);
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
template <interpolation::type IX, bound::type BX>
struct Multiscale<three, IX, BX, IX, BX, IX, BX> {
    using bound_utils_x = bound::utils<BX>;
    using inter_utils_x = interpolation::utils<IX>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, reduce_t wscl,
                reduce_t shift)
    {
        // Precompute weights and indices
        scalar_t x = wscl * (w + shift) - shift;
        offset_t bx0, bx1;
        inter_utils_x::bounds(x, bx0, bx1);
        offset_t dbx = bx1-bx0;
        scalar_t    wx[8];
        offset_t    ix[8];
        signed char sx[8];
        {
            scalar_t    *owx = static_cast<scalar_t*>(wx);
            offset_t    *oix = static_cast<offset_t*>(ix);
            signed char *osx = static_cast<signed char *>(sx);
            for (offset_t bx = bx0; bx <= bx1; ++bx) {
                scalar_t dx = x - bx;
                *(owx++)  = inter_utils_x::fastweight(dx);
                *(osx++)  = bound_utils_x::sign(bx, nw);
                *(oix++)  = bound_utils_x::index(bx, nw);
            }
        }

        // Convolve coefficients with basis functions
        scalar_t acc = static_cast<scalar_t>(0);
        for (offset_t i = 0; i <= dbx; ++i)
            acc += bound::get(inp, ix[i] * sw, sx[i]) * wx[i];
        *out = acc;

    }
};

/***********************************************************************
 *
 *                                  2D
 *
 **********************************************************************/

/***                           NEAREST                              ***/
template <bound::type BX, bound::type BY>
struct Multiscale<two, Z, BX, Z, BY, Z, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using inter_utils = interpolation::utils<Z>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, reduce_t wscl,
                offset_t h, offset_t nh, offset_t sh, reduce_t hscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        offset_t ix = static_cast<offset_t>(round(x));
        offset_t iy = static_cast<offset_t>(round(y));
        signed char  sx = bound_utils::sign(ix, nw);
        signed char  sy = bound_utils::sign(iy, nh);
        ix = bound_utils::index(ix, nw) * sw;
        iy = bound_utils::index(iy, nh) * sh;

        *out = bound::get(inp, ix + iy, sx * sy);
};

/***                            LINEAR                              ***/
template <bound::type BX, bound::type BY>
struct Multiscale<two, L, BX, L, BY, L, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using inter_utils = interpolation::utils<L>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, reduce_t wscl,
                offset_t h, offset_t nh, offset_t sh, reduce_t hscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        offset_t ix0 = static_cast<offset_t>(floor(x));
        offset_t iy0 = static_cast<offset_t>(floor(y));
        offset_t ix1 = ix0 + 1;
        offset_t iy1 = iy0 + 1;
        reduce_t dx1 = x - ix0;
        reduce_t dy1 = x - iy0;
        reduce_t dx0 = 1 - dx1;
        reduce_t dy0 = 1 - dy1;
        signed char  sx0 = bound_utils::sign(ix0, nw);
        signed char  sy0 = bound_utils::sign(iy0, nh);
        signed char  sx1 = bound_utils::sign(ix1, nw);
        signed char  sy1 = bound_utils::sign(iy1, nh);
        ix0 = bound_utils::index(ix0, nw) * sw;
        iy0 = bound_utils::index(iy0, nh) * sh;
        ix1 = bound_utils::index(ix1, nw) * sw;
        iy1 = bound_utils::index(iy1, nh) * sh;

        auto accum1d = [ix0, ix1, dx0, dx1, sx0, sx1, inp]
                        (offset_t i, signed char s)
        {
          return static_cast<reduce_t>(bound::get(inp, i + ix0, s * sx0)) * dx0
               + static_cast<reduce_t>(bound::get(inp, i + ix1, s * sx1)) * dx1;
        };

        *out = static_cast<scalar_t>(accum1d(iy0, sy0) * dy0
                                   + accum1d(iy1, sy1) * dy1);
};

/***                          QUADRATIC                             ***/
template <bound::type BX, bound::type BY>
struct Multiscale<two, Q, BX, Q, BY, Q, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using inter_utils = interpolation::utils<Q>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, reduce_t wscl,
                offset_t h, offset_t nh, offset_t sh, reduce_t hscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x+0.5));
        offset_t iy1 = static_cast<offset_t>(floor(y+0.5));
        reduce_t dx1 = inter_utils::weight(x - ix1);
        reduce_t dy1 = inter_utils::weight(y - iy1);
        reduce_t dx0 = inter_utils::fastweight(x - (ix1 - 1));
        reduce_t dy0 = inter_utils::fastweight(y - (iy1 - 1));
        reduce_t dx2 = inter_utils::fastweight((ix1 + 1) - x);
        reduce_t dy2 = inter_utils::fastweight((iy1 + 1) - y);
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
};

/***                            CUBIC                               ***/
template <bound::type BX, bound::type BY>
struct Multiscale<two, C, BX, C, BY, C, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using inter_utils = interpolation::utils<C>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, reduce_t wscl,
                offset_t h, offset_t nh, offset_t sh, reduce_t hscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x));
        offset_t iy1 = static_cast<offset_t>(floor(y));
        reduce_t dx1 = inter_utils::fastweight(x - ix1);
        reduce_t dy1 = inter_utils::fastweight(y - iy1);
        reduce_t dx0 = inter_utils::fastweight(x - (ix1 - 1));
        reduce_t dy0 = inter_utils::fastweight(y - (iy1 - 1));
        reduce_t dx2 = inter_utils::fastweight((ix1 + 1) - x);
        reduce_t dy2 = inter_utils::fastweight((iy1 + 1) - y);
        reduce_t dx3 = inter_utils::fastweight((ix1 + 2) - x);
        reduce_t dy3 = inter_utils::fastweight((iy1 + 2) - y);
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
};

/***                             ANY                                ***/
template <interpolation::type IX, bound::type BX,
          interpolation::type IY, bound::type BY>
struct Multiscale<three, IX, BX, IY, BY, IY, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using inter_utils_x = interpolation::utils<IX>;
    using inter_utils_y = interpolation::utils<IY>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, reduce_t wscl,
                offset_t h, offset_t nh, offset_t sh, reduce_t hscl,
                reduce_t shift)
    {
        // Precompute weights and indices
        scalar_t x = wscl * (w + shift) - shift;
        scalar_t y = hscl * (h + shift) - shift;
        offset_t bx0, bx1, by0, by1;
        inter_utils_x::bounds(x, bx0, bx1);
        inter_utils_y::bounds(y, by0, by1);
        offset_t dbx = bx1-bx0;
        offset_t dby = by1-by0;
        scalar_t    wx[8],  wy[8];
        offset_t    ix[8],  iy[8];
        signed char sx[8],  sy[8];
        {
            scalar_t    *owy = static_cast<scalar_t*>(wy);
            offset_t    *oiy = static_cast<offset_t*>(iy);
            signed char *osy = static_cast<signed char *>(sy);
            for (offset_t by = by0; by <= by1; ++by) {
                scalar_t dy = y - by;
                *(owy++)  = inter_utils_y::fastweight(dy);
                *(osy++)  = bound_utils_y::sign(by, nh);
                *(oiy++)  = bound_utils_y::index(by, nh);
            }
        }
        {
            scalar_t    *owx = static_cast<scalar_t*>(wx);
            offset_t    *oix = static_cast<offset_t*>(ix);
            signed char *osx = static_cast<signed char *>(sx);
            for (offset_t bx = bx0; bx <= bx1; ++bx) {
                scalar_t dx = x - bx;
                *(owx++)  = inter_utils_x::fastweight(dx);
                *(osx++)  = bound_utils_x::sign(bx, nw);
                *(oix++)  = bound_utils_x::index(bx, nw);
            }
        }

        // Convolve coefficients with basis functions
        scalar_t acc = static_cast<scalar_t>(0);
        for (offset_t j = 0; j <= dby; ++j) {
            offset_t    osy = iy[j] * sh;
            signed char syy = sy[j];
            scalar_t    wyy = wy[j];
            for (offset_t i = 0; i <= dbx; ++i) {
                offset_t    osxy = osyy + ix[i] * sw;
                signed char sxy  = syy  * sx[i];
                scalar_t    wxy  = wyy  * wx[i];
                acc += bound::get(inp, osxy, sxy) * wxy;
            }
        }
        *out = acc;

    }
};

/***********************************************************************
 *
 *                                  3D
 *
 **********************************************************************/

/***                          QUADRATIC                             ***/
template <bound::type BX, bound::type BY, bound::type BZ>
struct Multiscale<three, Q, BX, Q, BY, Q, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using inter_utils = interpolation::utils<Q>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, reduce_t wscl,
                offset_t h, offset_t nh, offset_t sh, reduce_t hscl,
                offset_t d, offset_t nd, offset_t sd, reduce_t dscl,
                reduce_t shift)
    {
        reduce_t x = (w + shift) * wscl - shift;
        reduce_t y = (h + shift) * hscl - shift;
        reduce_t z = (d + shift) * dscl - shift;
        offset_t ix1 = static_cast<offset_t>(floor(x+0.5));
        offset_t iy1 = static_cast<offset_t>(floor(y+0.5));
        offset_t iz1 = static_cast<offset_t>(floor(z+0.5));
        reduce_t dx1 = inter_utils::weight(x - ix1);
        reduce_t dy1 = inter_utils::weight(y - iy1);
        reduce_t dz1 = inter_utils::weight(z - iz1);
        reduce_t dx0 = inter_utils::fastweight(x - (ix1 - 1));
        reduce_t dy0 = inter_utils::fastweight(y - (iy1 - 1));
        reduce_t dz0 = inter_utils::fastweight(z - (iz1 - 1));
        reduce_t dx2 = inter_utils::fastweight((ix1 + 1) - x);
        reduce_t dy2 = inter_utils::fastweight((iy1 + 1) - y);
        reduce_t dz2 = inter_utils::fastweight((iz1 + 1) - z);
        int8_t  sx0 = bound_utils_x::sign(ix1-1, nw);
        int8_t  sy0 = bound_utils_y::sign(iy1-1, nh);
        int8_t  sz0 = bound_utils_z::sign(iz1-1, nd);
        int8_t  sx2 = bound_utils_x::sign(ix1+1, nw);
        int8_t  sy2 = bound_utils_y::sign(iy1+1, nh);
        int8_t  sz2 = bound_utils_z::sign(iz1+1, nd);
        int8_t  sx1 = bound_utils_x::sign(ix1,   nw);
        int8_t  sy1 = bound_utils_y::sign(iy1,   nh);
        int8_t  sz1 = bound_utils_z::sign(iz1,   nd);
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
};

/***                             ANY                                ***/
template <interpolation::type IX, bound::type BX,
          interpolation::type IY, bound::type BY,
          interpolation::type IZ, bound::type BZ>
struct Multiscale<three, IX, BX, IY, BY, IZ, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    using inter_utils_x = interpolation::utils<IX>;
    using inter_utils_y = interpolation::utils<IY>;
    using inter_utils_z = interpolation::utils<IZ>;

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                offset_t w, offset_t nw, offset_t sw, reduce_t wscl,
                offset_t h, offset_t nh, offset_t sh, reduce_t hscl,
                offset_t d, offset_t nd, offset_t sd, reduce_t dscl,
                reduce_t shift)
    {
        // Precompute weights and indices
        scalar_t x = wscl * (w + shift) - shift;
        scalar_t y = hscl * (h + shift) - shift;
        scalar_t z = dscl * (d + shift) - shift;
        offset_t bx0, bx1, by0, by1, bz0, bz1;
        inter_utils_x::bounds(x, bx0, bx1);
        inter_utils_y::bounds(y, by0, by1);
        inter_utils_z::bounds(z, bz0, bz1);
        offset_t dbx = bx1-bx0;
        offset_t dby = by1-by0;
        offset_t dbz = bz1-bz0;
        scalar_t    wx[8],  wy[8],  wz[8];
        offset_t    ix[8],  iy[8],  iz[8];
        signed char sx[8],  sy[8],  sz[8];
        {
            scalar_t    *owz = wz;
            offset_t    *oiz = iz;
            signed char *osz = sz;
            for (offset_t bz = bz0; bz <= bz1; ++bz) {
                scalar_t dz = z - bz;
                *(owz++)  = inter_utils_z::fastweight(dz);
                *(osz++)  = bound_utils_z::sign(bz, nd);
                *(oiz++)  = bound_utils_z::index(bz, nd);
            }
        }
        {
            scalar_t    *owy = wy;
            offset_t    *oiy = iy;
            signed char *osy = sy;
            for (offset_t by = by0; by <= by1; ++by) {
                scalar_t dy = y - by;
                *(owy++)  = inter_utils_y::fastweight(dy);
                *(osy++)  = bound_utils_y::sign(by, nh);
                *(oiy++)  = bound_utils_y::index(by, nh);
            }
        }
        {
            scalar_t    *owx = wx;
            offset_t    *oix = ix;
            signed char *osx = sx;
            for (offset_t bx = bx0; bx <= bx1; ++bx) {
                scalar_t dx = x - bx;
                *(owx++)  = inter_utils_x::fastweight(dx);
                *(osx++)  = bound_utils_x::sign(bx, nw);
                *(oix++)  = bound_utils_x::index(bx, nw);
            }
        }

        // Convolve coefficients with basis functions
        scalar_t acc = static_cast<scalar_t>(0);
        for (offset_t k = 0; k <= dbz; ++k) {
            offset_t    osz = iz[k] * sd;
            signed char szz = sz[k];
            scalar_t    wzz = wz[k];
            for (offset_t j = 0; j <= dby; ++j) {
                offset_t    osyz = osz + iy[j] * sh;
                signed char syz  = szz * sy[j];
                scalar_t    wyz  = wzz * wy[j];
                for (offset_t i = 0; i <= dbx; ++i) {
                    offset_t    osxyz = osyz + ix[i] * sw;
                    signed char sxyz  = syz  * sx[i];
                    scalar_t    wxyz  = wyz  * wx[i];
                    acc += bound::get(inp, osxyz, sxyz) * wxyz;
                }
            }
        }
        *out = acc;

    }
};

/***********************************************************************
 *
 *                                  ND
 *
 **********************************************************************/

template <int D>
struct Multiscale<D, Z, B0, Z, B0, Z, B0> {

    template <typename scalar_t, typename offset_t, typename reduce_t>
    static __device__
    void resize(scalar_t * out, scalar_t * inp,
                const offset_t * coord, const offset_t * size, const offset_t * stride,
                const interpolation::type * inter, const * bound::type bnd,
                const reduce_t * scl, reduce_t shift)
    {
        // Precompute weights and indices
        scalar_t    w[8*D];
        offset_t    i[8*D];
        signed char s[8*D];
        offset_t    db[D];
        for (offset_t d=0; d<D; ++d) {
            scalar_t    *wd = w + 8*d;
            offset_t    *id = i + 8*d;
            signed char *sd = s + 8*d;
            scalar_t x = scl[d] * (coord[d] + shift) - shift;
            offset_t b0, b1;
            interpolation::bounds(inter[d], x, b0, b1);
            db[d] = b1-b0;
            for (offset_t b = b0; b <= b1; ++b) {
                *(wd++) = interpolation::fastweight(inter[d], x - b);
                *(sd++) = bound::sign(bnd[d], b, size[d]);
                *(id++) = bound::index(bnd[d], b, size[d]);
            }
        }

        // Convolve coefficients with basis functions
        offset_t    offsets[D];
        signed char signs[D];
        scalar_t    weights[D];
        scalar_t acc = static_cast<scalar_t>(0);
        for (offset_t d=0; d<D; ++d) {
            scalar_t    *wd = w + 8*d;
            offset_t    *id = i + 8*d;
            signed char *sd = s + 8*d;
            for (offset_t k = 0; k <= db[d]; ++k) {
                offsets[d] = (d > 0 ? offsets[d-1] : static_cast<offset_t>(0))
                           + id[k] * stride[d];
                signs[d]   = (d > 0 ? signs[d-1]   : static_cast<signed char>(1))
                           * sd[k];
                weights[d] = (d > 0 ? weights[d-1] : static_cast<scalar_t>(1))
                           * wd[k];
                if (d == D-1)
                    acc += bound::get(inp, offsets[D-1], signs[D-1]) * weights[D-1];
            }
        }
        *out = acc;
    }
};


/***********************************************************************
 *
 *                              KERNELS
 *
 **********************************************************************/

template <interpolation::type IX, bound::type BX,
          typename scalar_t, typename offset_t>
__global__ void kernel1d(scalar_t * out, scalar_t * inp, int ndim,
                         scalar_t shift, const scalar_t * scale,
                         const offset_t * size_out,
                         const offset_t * size_inp,
                         const offset_t * stride_out,
                         const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x;
        offset_t batch_offset = index2offset_1d(i, ndim, size_out, stride_inp, x);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<one, IX, BX>::resize(out + out_offset, inp + batch_offset,
                                        x, size_inp[ndim-1], stride_inp[ndim-1],
                                        scale[ndim-1], shift);
    }
}

template <interpolation::type IX, bound::type BX,
          interpolation::type IY, bound::type BY,
          typename scalar_t, typename offset_t>
__global__ void kernel2d(scalar_t * out, scalar_t * inp, int ndim,
                         scalar_t shift, const scalar_t * scale,
                         const offset_t * size_out,
                         const offset_t * size_inp,
                         const offset_t * stride_out,
                         const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x, y;
        offset_t batch_offset = index2offset_2d(i, ndim, size_out, stride_inp, x, y);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<one, IX, BX, IY, BY>::resize(
            out + out_offset, inp + batch_offset,
            x, size_inp[ndim-2], stride_inp[ndim-2], scale[ndim-2],
            y, size_inp[ndim-1], stride_inp[ndim-1], scale[ndim-1],
            shift);
    }
}

template <interpolation::type IX, bound::type BX,
          interpolation::type IY, bound::type BY,
          interpolation::type IZ, bound::type BZ,
          typename scalar_t, typename offset_t>
__global__ void kernel3d(scalar_t * out, scalar_t * inp, int ndim,
                         scalar_t shift, const scalar_t * scale,
                         const offset_t * size_out,
                         const offset_t * size_inp,
                         const offset_t * stride_out,
                         const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x, y, z;
        offset_t batch_offset = index2offset_3d(i, ndim, size_out, stride_inp, x, y, z);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<one, IX, BX, IY, BY, IZ, BZ>::resize(
            out + out_offset, inp + batch_offset,
            x, size_inp[ndim-3], stride_inp[ndim-3], scale[ndim-3],
            y, size_inp[ndim-2], stride_inp[ndim-2], scale[ndim-2],
            z, size_inp[ndim-1], stride_inp[ndim-1], scale[ndim-1],
            shift);
    }
}

template <int D, typename scalar_t, typename offset_t>
__global__ void kernelnd(scalar_t * out, scalar_t * inp, int ndim,
                         scalar_t shift, const scalar_t * scale,
                         const interpolation::type * order,
                         const bound::type * bnd,
                         const offset_t * size_out,
                         const offset_t * size_inp,
                         const offset_t * stride_out,
                         const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nthreads = prod(size_out, ndim);

    for (offset_t i=index; index < nthreads;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t x[D];
        offset_t batch_offset = index2offset_nd(i, ndim, size_out, stride_inp, x, D);
        offset_t out_offset = index2offset(i, ndim, size_out, stride_out);

        Multiscale<D>::resize(
            out + out_offset, inp + batch_offset,
            x, size_inp + ndim - D, stride_inp + ndim - D,
            order, bnd, scale, shift);
    }
}
