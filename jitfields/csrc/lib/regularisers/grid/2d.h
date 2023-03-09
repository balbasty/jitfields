#ifndef JF_REGULARISERS_GRID_2D
#define JF_REGULARISERS_GRID_2D
#include "../../cuda_switch.h"
#include "../../bounds.h"
#include "../../utils.h"
#include "utils.h"

namespace jf {
namespace reg_grid {

//----------------------------------------------------------------------
//          low-level kernels for anything regularization
//----------------------------------------------------------------------

template <typename scalar_t, typename reduce_t, typename offset_t,
          bound::type BX, bound::type BY>
struct RegGrid<two, scalar_t, reduce_t, offset_t, BX, BY> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    typedef scalar_t & (*OpType)(scalar_t &, const reduce_t &);

    //------------------------------------------------------------------
    //                            ABSOLUTE
    //------------------------------------------------------------------

    static const int kernelsize_absolute = 2;

    /// kernel <- [absx, absy]
    __device__ static inline void
    make_kernel_absolute(
        reduce_t * kernel, reduce_t absolute, const reduce_t voxel_size[2])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);
        kernel[0] = absolute / vx;
        kernel[1] = absolute / vy;
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_absolute(
        scalar_t * out, const scalar_t * inp,
        offset_t osc, offset_t isc, const reduce_t kernel[2])
    {
        op(out[0],     kernel[0] * inp[0]);
        op(out[osc],   kernel[1] * inp[isc]);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline  void
    kernel_absolute(scalar_t * out, offset_t osc, const reduce_t kernel[2])
    {
        op(out[0],     kernel[0]);
        op(out[osc],   kernel[1]);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline  void
    diag_absolute(scalar_t * out, offset_t osc, const reduce_t kernel[2])
    {
        return kernel_absolute(out, osc, kernel);
    }

    //------------------------------------------------------------------
    //                            MEMBRANE
    //------------------------------------------------------------------

    static const int kernelsize_membrane = 6;

    /// kernel <- [absx, wx10, wx01,
    ///            absy, wy10, wy01]
    __device__ static inline void
    make_kernel_membrane(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);
        kernel[0]  = absolute / vx;
        kernel[1]  = -membrane;
        kernel[2]  = -membrane * (vy/vx);
        kernel[3]  = absolute / vy;
        kernel[4]  = -membrane * (vx/vy);
        kernel[5]  = -membrane;
    }

    /// kernel <- [wx00, wx10, wx01,
    ///            wy00, wy10, wy01]
    __device__ static inline void
    make_fullkernel_membrane(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[2])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);
        kernel[0]  = (absolute + 2 * membrane * (vx + vy)) / vx;
        kernel[1]  = -membrane;
        kernel[2]  = -membrane * (vy/vx);
        kernel[3]  = (absolute + 2 * membrane * (vx + vy)) / vy;
        kernel[4]  = -membrane * (vx/vy);
        kernel[5]  = -membrane;
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_membrane(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[2], const offset_t size[2], const offset_t stride[2],
        offset_t osc, offset_t isc, const reduce_t kernel[6])
    {
        offset_t  x = loc[0],     y = loc[1];
        offset_t nx = size[0],   ny = size[1];
        offset_t sx = stride[0], sy = stride[1];

        offset_t x0 = x-1, x1 = x+1, y0 = y-1, y1 = y+1;
        signed char fx0 = bound_utils_x::sign(x0, nx);
        signed char fx1 = bound_utils_x::sign(x1, nx);
        signed char fy0 = bound_utils_y::sign(y0, ny);
        signed char fy1 = bound_utils_y::sign(y1, ny);
        x0 = (bound_utils_x::index(x0, nx) - x) * sx;
        x1 = (bound_utils_x::index(x1, nx) - x) * sx;
        y0 = (bound_utils_y::index(y0, ny) - y) * sy;
        y1 = (bound_utils_y::index(y1, ny) - y) * sy;

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t center = static_cast<reduce_t>(inp[0]);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out, kernel[0] * center +
                     kernel[1] * (get(x0, fx0) + get(x1, fx1)) +
                     kernel[2] * (get(y0, fy0) + get(y1, fy1)));
        };

        conv(out,         inp,         kernel);
        conv(out + osc,   inp + isc,   kernel + 3);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
    kernel_membrane(
        scalar_t * out, offset_t sc, const offset_t stride[2],
        const reduce_t kernel[6])
    {
        offset_t sx = stride[0], sy = stride[1];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0], w100 = kernel[1], w010 = kernel[2];
            op(out[0],   w000);
            op(out[-sx], w100);
            op(out[+sx], w100);
            op(out[-sy], w010);
            op(out[+sy], w010);
        };

        setkernel(out,        kernel);
        setkernel(out + sc,   kernel + 3);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_membrane(
        scalar_t * out, offset_t osc,
         const offset_t loc[2], const offset_t size[2],
         const reduce_t kernel[6])
    {
        offset_t  x = loc[0],   y = loc[1];
        offset_t nx = size[0], ny = size[1];

        signed char fx = bound_utils_x::sign(x-1, nx)
                       + bound_utils_x::sign(x+1, nx);
        signed char fy = bound_utils_y::sign(y-1, ny)
                       + bound_utils_y::sign(y+1, ny);

         op(out[0],     kernel[0] - kernel[1]*fx - kernel[2]*fy);
         op(out[osc],   kernel[3] - kernel[4]*fx - kernel[5]*fy);
    }

    //------------------------------------------------------------------
    //                            BENDING
    //------------------------------------------------------------------

    static const int kernelsize_bending = 12;

    /// kernel <- [
    ///     absx, wx100, wx010, wx200, wx020, wx110,
    ///     absy, wy100, wy010, wy200, wy020, wy110]
    __device__ static inline void
    make_kernel_bending(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        const reduce_t voxel_size[2])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);

        reduce_t w100 = -4 * bending * vx * (vx + vy) - membrane * vx;
        reduce_t w010 = -4 * bending * vy * (vx + vy) - membrane * vy;
        reduce_t w200 = bending * vx * vx;
        reduce_t w020 = bending * vy * vy;
        reduce_t w110 = 2 * bending * vx * vy;
        reduce_t w000 = absolute;

        kernel[0]  = w000 / vx;
        kernel[1]  = w100 / vx;
        kernel[2]  = w010 / vx;
        kernel[3]  = w200 / vx;
        kernel[4]  = w020 / vx;
        kernel[5]  = w110 / vx;

        kernel[6]  = w000 / vy;
        kernel[7]  = w100 / vy;
        kernel[8]  = w010 / vy;
        kernel[9]  = w200 / vy;
        kernel[10] = w020 / vy;
        kernel[11] = w110 / vy;
    }

    /// kernel <- [
    ///     wx000, wx100, wx010, wx200, wx020, wx110,
    ///     wy000, wy100, wy010, wy200, wy020, wy110]
    __device__ static inline void
    make_fullkernel_bending(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        const reduce_t voxel_size[2])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);

        reduce_t w100 = -4 * bending * vx * (vx + vy) - membrane * vx;
        reduce_t w010 = -4 * bending * vy * (vx + vy) - membrane * vy;
        reduce_t w200 = bending * vx * vx;
        reduce_t w020 = bending * vy * vy;
        reduce_t w110 = 2 * bending * vx * vy;
        reduce_t w000 = absolute
                      - 2 * (w100 + w010 + w200 + w020)
                      - 4 * (w110);

        kernel[0]  = w000 / vx;
        kernel[1]  = w100 / vx;
        kernel[2]  = w010 / vx;
        kernel[3]  = w200 / vx;
        kernel[4]  = w020 / vx;
        kernel[5]  = w110 / vx;

        kernel[6]  = w000 / vy;
        kernel[7]  = w100 / vy;
        kernel[8]  = w010 / vy;
        kernel[9]  = w200 / vy;
        kernel[10] = w020 / vy;
        kernel[11] = w110 / vy;
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_bending(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[2], const offset_t size[2],
        const offset_t stride[2], offset_t osc, offset_t isc,
        const reduce_t kernel[12])
    {
        offset_t  x = loc[0],     y = loc[1];
        offset_t nx = size[0],   ny = size[1];
        offset_t sx = stride[0], sy = stride[1];

        signed char fx00 = bound_utils_x::sign(x-2, nx);
        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        signed char fx11 = bound_utils_x::sign(x+2, nx);
        signed char fy00 = bound_utils_y::sign(y-2, ny);
        signed char fy0  = bound_utils_y::sign(y-1, ny);
        signed char fy1  = bound_utils_y::sign(y+1, ny);
        signed char fy11 = bound_utils_y::sign(y+2, ny);
        offset_t    x00 = (bound_utils_x::index(x-2, nx) - x) * sx;
        offset_t    x0  = (bound_utils_x::index(x-1, nx) - x) * sx;
        offset_t    x1  = (bound_utils_x::index(x+1, nx) - x) * sx;
        offset_t    x11 = (bound_utils_x::index(x+2, nx) - x) * sx;
        offset_t    y00 = (bound_utils_y::index(y-2, ny) - y) * sy;
        offset_t    y0  = (bound_utils_y::index(y-1, ny) - y) * sy;
        offset_t    y1  = (bound_utils_y::index(y+1, ny) - y) * sy;
        offset_t    y11 = (bound_utils_y::index(y+2, ny) - y) * sy;

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1], w010 = kernel[2],
                     w200 = kernel[3], w020 = kernel[4],
                     w110 = kernel[5];

            reduce_t center = static_cast<reduce_t>(inp[0]);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
                  w000 * center
                + w100 * (get(x0, fx0) + get(x1, fx1))
                + w010 * (get(y0, fy0) + get(y1, fy1))
                + w200 * (get(x00, fx00) + get(x11, fx11))
                + w020 * (get(y00, fy00) + get(y11, fy11))
                + w110 * (get(x0+y0, fx0*fy0) + get(x1+y0, fx1*fy0) +
                          get(x0+y1, fx0*fy1) + get(x1+y1, fx1*fy1))
            );
        };

        conv(out,         inp,         kernel);
        conv(out + osc,   inp + isc,   kernel + 6);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
     kernel_bending(
        scalar_t * out, offset_t sc, const offset_t stride[2],
        const reduce_t kernel[12])
    {
        offset_t sx = stride[0], sy = stride[1];

        auto setkernel = [&](scalar_t * o, const reduce_t * ker) {
            reduce_t w000 = ker[0],
                     w100 = ker[1], w010 = ker[2],
                     w200 = ker[3], w020 = ker[4],
                     w110 = ker[5];
            op(o[0],      w000);
            op(o[-sx],    w100);
            op(o[+sx],    w100);
            op(o[-sy],    w010);
            op(o[+sy],    w010);
            op(o[-sx*2],  w200);
            op(o[+sx*2],  w200);
            op(o[-sy*2],  w020);
            op(o[+sy*2],  w020);
            op(o[-sx-sy], w110);
            op(o[-sx+sy], w110);
            op(o[+sx-sy], w110);
            op(o[+sx+sy], w110);
        };

        setkernel(out,        kernel);
        setkernel(out + sc,   kernel + 6);
    }

    // --- diagonal ---

    template <OpType op = set>
    static inline __device__ void
    diag_bending(
        scalar_t * out, offset_t osc,
        const offset_t loc[2], const offset_t size[2],
        const reduce_t kernel[12])
    {
        offset_t  x = loc[0],     y = loc[1];
        offset_t nx = size[0],   ny = size[1];

        signed char fx00 = bound_utils_x::sign(x-2, nx);
        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        signed char fx11 = bound_utils_x::sign(x+2, nx);
        signed char fy00 = bound_utils_y::sign(y-2, ny);
        signed char fy0  = bound_utils_y::sign(y-1, ny);
        signed char fy1  = bound_utils_y::sign(y+1, ny);
        signed char fy11 = bound_utils_y::sign(y+2, ny);

        auto setdiag = [&](scalar_t & out, const reduce_t * kernel) {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1], w010 = kernel[2],
                     w200 = kernel[3], w020 = kernel[4],
                     w110 = kernel[5];
            w000 -=   w100 * (fx0 + fx1)   + w010 * (fy0 + fy1)
                    + w200 * (fx00 + fx11) + w020 * (fy00 + fy11)
                    + w110 * (fx0*fy0 + fx1*fy0 + fx1*fy0 + fx1*fy1);
            op(out, w000);
        };
        setdiag(out[0],     kernel);
        setdiag(out[osc],   kernel + 6);
    }

    //------------------------------------------------------------------
    //                          LAME + BENDING
    //------------------------------------------------------------------

    static const int kernelsize_all = 13;

    /// kernel <- [
    ///      absx, wx100, wx010, wx200, wx020, wx110,
    ///      absy, wy100, wy010, wy200, wy020, wy110,
    ///      ww]
    static inline __device__ void
    make_kernel_all(
        reduce_t * kernel,
        reduce_t absolute, reduce_t membrane, reduce_t bending,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[2])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);
        reduce_t vxy = vx + vy;

        reduce_t w100 = (-4 * bending * vxy - membrane) * vx;
        reduce_t w010 = (-4 * bending * vxy - membrane) * vy;
        reduce_t w200 = bending * vx * vx;
        reduce_t w020 = bending * vy * vy;
        reduce_t w110 = 2 * bending * vx * vy;
        reduce_t  w000 = absolute;

        kernel[0]  = w000 / vx;
        kernel[1]  = w100 / vx - 2*shears - div;
        kernel[2]  = w010 / vx - shears*(vy/vx);
        kernel[3]  = w200 / vx;
        kernel[4]  = w020 / vx;
        kernel[5]  = w110 / vx;

        kernel[6]  = w000 / vy;
        kernel[7]  = w100 / vy - shears*(vx/vy);
        kernel[8]  = w010 / vy - 2*shears - div;
        kernel[9]  = w200 / vy;
        kernel[10] = w020 / vy;
        kernel[11] = w110 / vy;

        kernel[12] = 0.25 * (shears + div);
    }

    /// kernel <- [
    ///      wx000, wx100, wx010, wx200, wx020, wx110,
    ///      wy000, wy100, wy010, wy200, wy020, wy110,
    ///      ww]
    static inline __device__ void make_fullkernel_all(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        reduce_t shears, reduce_t div,
        const reduce_t voxel_size[2])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);
        reduce_t vxy = vx + vy;

        reduce_t w100 = (-4 * bending * vxy - membrane) * vx;
        reduce_t w010 = (-4 * bending * vxy - membrane) * vy;
        reduce_t w200 = bending * vx * vx;
        reduce_t w020 = bending * vy * vy;
        reduce_t w110 = 2 * bending * vx * vy;
        reduce_t w000 = absolute
                      - 2 * (w100 + w010 + w200 + w020)
                      - 4 * (w110);

        kernel[0]  = w000 / vx + 2*shears*(2*vx+vy)/vx + 2*div;
        kernel[1]  = w100 / vx - 2*shears - div;
        kernel[2]  = w010 / vx - shears*(vy/vx);
        kernel[3]  = w200 / vx;
        kernel[4]  = w020 / vx;
        kernel[5]  = w110 / vx;

        kernel[6]  = w000 / vy + 2*shears*(vx+2*vy)/vy + 2*div;
        kernel[7]  = w100 / vy - shears*(vx/vy);
        kernel[8]  = w010 / vy - 2*shears - div;
        kernel[9]  = w200 / vy;
        kernel[10] = w020 / vy;
        kernel[11] = w110 / vy;

        kernel[12] = 0.25 * (shears + div);
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_all(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[2], const offset_t size[2],
        const offset_t stride[2], offset_t osc, offset_t isc,
        const reduce_t kernel[13])
    {
        offset_t  x = loc[0],     y = loc[1];
        offset_t nx = size[0],   ny = size[1];
        offset_t sx = stride[0], sy = stride[1];

        signed char fx00 = bound_utils_x::sign(x-2, nx);
        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        signed char fx11 = bound_utils_x::sign(x+2, nx);
        signed char fy00 = bound_utils_y::sign(y-2, ny);
        signed char fy0  = bound_utils_y::sign(y-1, ny);
        signed char fy1  = bound_utils_y::sign(y+1, ny);
        signed char fy11 = bound_utils_y::sign(y+2, ny);
        offset_t    x00 = (bound_utils_x::index(x-2, nx) - x) * sx;
        offset_t    x0  = (bound_utils_x::index(x-1, nx) - x) * sx;
        offset_t    x1  = (bound_utils_x::index(x+1, nx) - x) * sx;
        offset_t    x11 = (bound_utils_x::index(x+2, nx) - x) * sx;
        offset_t    y00 = (bound_utils_y::index(y-2, ny) - y) * sy;
        offset_t    y0  = (bound_utils_y::index(y-1, ny) - y) * sy;
        offset_t    y1  = (bound_utils_y::index(y+1, ny) - y) * sy;
        offset_t    y11 = (bound_utils_y::index(y+2, ny) - y) * sy;

        reduce_t center0 = static_cast<reduce_t>(inp[0]),
                 center1 = static_cast<reduce_t>(inp[isc]);

        auto cget0 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp, o, f) - center0;
        };
        auto get0 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp, o, f);
        };
        auto cget1 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc, o, f) - center1;
        };
        auto get1 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc, o, f);
        };

        auto w2 = kernel[12];

        {
            reduce_t wx000 = kernel[0],
                     wx100 = kernel[1], wx010 = kernel[2],
                     wx200 = kernel[3], wx020 = kernel[4],
                     wx110 = kernel[5];

            op(out[0],
                  wx000 * center0
                + wx100 * (cget0(x0, fx0)        + cget0(x1, fx1))
                + wx010 * (cget0(y0, fy0)        + cget0(y1, fy1))
                + wx200 * (cget0(x00, fx00)      + cget0(x11, fx11))
                + wx020 * (cget0(y00, fy00)      + cget0(y11, fy11))
                + wx110 * (cget0(x0+y0, fx0*fy0) + cget0(x1+y0, fx1*fy0) +
                           cget0(x0+y1, fx0*fy1) + cget0(x1+y1, fx1*fy1))
                + w2 * (
                      get1(x1+y0, fx1*fy0) + get1(x0+y1, fx0*fy1)
                    - get1(x0+y0, fx1*fy1) - get1(x1+y1, fx1*fy1)
                )
            );
        }

        kernel += 6;

        {
            reduce_t wy000 = kernel[0],
                     wy100 = kernel[1], wy010 = kernel[2],
                     wy200 = kernel[3], wy020 = kernel[4],
                     wy110 = kernel[5];

            op(out[osc],
                  wy000 * center1
                + wy100 * (cget1(x0, fx0)        + cget1(x1, fx1))
                + wy010 * (cget1(y0, fy0)        + cget1(y1, fy1))
                + wy200 * (cget1(x00, fx00)      + cget1(x11, fx11))
                + wy020 * (cget1(y00, fy00)      + cget1(y11, fy11))
                + wy110 * (cget1(x0+y0, fx0*fy0) + cget1(x1+y0, fx1*fy0) +
                           cget1(x0+y1, fx0*fy1) + cget1(x1+y1, fx1*fy1))
                + w2 * (
                      get0(x1+y0, fx1*fy0) + get0(x0+y1, fx0*fy1)
                    - get0(x0+y0, fx1*fy1) - get0(x1+y1, fx1*fy1)
                )
            );
        }
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
    kernel_all(
        scalar_t * out, const offset_t sc[2],
        const offset_t stride[2], const reduce_t kernel[13])
    {
        reduce_t sc0 = sc[0], sc1 = sc[1];
        offset_t sx = stride[0], sy = stride[1];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1], w010 = kernel[2],
                     w200 = kernel[3], w020 = kernel[4],
                     w110 = kernel[5];
            op(out[0],      w000);
            op(out[-sx],    w100);
            op(out[+sx],    w100);
            op(out[-sy],    w010);
            op(out[+sy],    w010);
            op(out[-sx*2],  w200);
            op(out[+sx*2],  w200);
            op(out[-sy*2],  w020);
            op(out[+sy*2],  w020);
            op(out[-sx-sy], w110);
            op(out[-sx+sy], w110);
            op(out[+sx-sy], w110);
            op(out[+sx+sy], w110);
        };

        auto xxout = out, yyout = out + sc0 + sc1;
        setkernel(xxout, kernel);
        setkernel(yyout, kernel + 6);

        auto w2 = kernel[12];
        auto xyout = out + sc0, yxout = out + sc1;
        op(xyout[+sx+sy], -w2); op(xyout[-sx-sy], -w2); op(yxout[+sx+sy], -w2); op(yxout[-sx-sy], -w2);
        op(xyout[-sx+sy], +w2); op(xyout[+sx-sy], +w2); op(yxout[-sx+sy], +w2); op(yxout[+sx-sy], +w2);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline  void
    diag_all(
        scalar_t * out, offset_t osc,
        const offset_t loc[2], const offset_t size[2],
        const reduce_t kernel[13])
    {
        offset_t  x = loc[0],   y = loc[1];
        offset_t nx = size[0], ny = size[1];

        signed char fx00 = bound_utils_x::sign(x-2, nx);
        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        signed char fx11 = bound_utils_x::sign(x+2, nx);
        signed char fy00 = bound_utils_y::sign(y-2, ny);
        signed char fy0  = bound_utils_y::sign(y-1, ny);
        signed char fy1  = bound_utils_y::sign(y+1, ny);
        signed char fy11 = bound_utils_y::sign(y+2, ny);

        reduce_t w2 = kernel[12];

        auto setdiag = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1], w010 = kernel[2],
                     w200 = kernel[3], w020 = kernel[4],
                     w110 = kernel[5];
            w000 -=   w100 * (fx0 + fx1)   + w010 * (fy0 + fy1)
                    + w200 * (fx00 + fx11) + w020 * (fy00 + fy11)
                    + w110 * (fx0*fy0 + fx1*fy0 + fx1*fy0 + fx1*fy1);
            op(*out, w000);
        };

        setdiag(out,         kernel);
        setdiag(out + osc,   kernel + 6);
    }

    //------------------------------------------------------------------
    //                          LAME
    //------------------------------------------------------------------

    static const int kernelsize_lame = 7;

    /// kernel <- [
    ///      absx, wx100, wx010,
    ///      absy, wy100, wy010,
    ///      ww]
    __device__ static inline  void
    make_kernel_lame(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[2])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);

        reduce_t w100 = - membrane * vx;
        reduce_t w010 = - membrane * vy;
        reduce_t w000 = absolute;

        kernel[0] = w000 / vx;
        kernel[1] = w100 / vx - 2*shears - div;
        kernel[2] = w010 / vx - shears*(vy/vx);

        kernel[3] = w000 / vy;
        kernel[4] = w100 / vy - shears*(vx/vy);
        kernel[5] = w010 / vy - 2*shears - div;

        kernel[6] = 0.25 * (shears + div);
    }

    /// kernel <- [
    ///      wx000, wx100, wx010,
    ///      wy000, wy100, wy010,
    ///      ww]
    __device__ static inline  void
    make_fullkernel_lame(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);

        reduce_t w100 = - membrane * vx;
        reduce_t w010 = - membrane * vy;
        reduce_t w000 = absolute - 2 * (w100 + w010);

        kernel[0] = w000 / vx + 2*shears*(2*vx+vy)/vx + 2*div;
        kernel[1] = w100 / vx - 2*shears - div;
        kernel[2] = w010 / vx - shears*(vy/vx);

        kernel[3] = w000 / vy + 2*shears*(vx+2*vy)/vy + 2*div;
        kernel[4] = w100 / vy - shears*(vx/vy);
        kernel[5] = w010 / vy - 2*shears - div;

        kernel[6] = 0.25 * (shears + div);
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_lame(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[2], const offset_t size[2],
        const offset_t stride[2], offset_t osc, offset_t isc,
        const reduce_t kernel[7])
    {
        offset_t  x = loc[0],     y = loc[1];
        offset_t nx = size[0],   ny = size[1];
        offset_t sx = stride[0], sy = stride[1];

        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        signed char fy0  = bound_utils_y::sign(y-1, ny);
        signed char fy1  = bound_utils_y::sign(y+1, ny);
        offset_t    x0  = (bound_utils_x::index(x-1, nx) - x) * sx;
        offset_t    x1  = (bound_utils_x::index(x+1, nx) - x) * sx;
        offset_t    y0  = (bound_utils_y::index(y-1, ny) - y) * sy;
        offset_t    y1  = (bound_utils_y::index(y+1, ny) - y) * sy;

        reduce_t wx000 = kernel[0], wx100 = kernel[1], wx010 = kernel[2],
                 wy000 = kernel[3], wy100 = kernel[4], wy010 = kernel[5],
                 w2    = kernel[6];

        reduce_t center0 = static_cast<reduce_t>(inp[0]),
                 center1 = static_cast<reduce_t>(inp[isc]);

        auto cget0 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp, o, f) - center0;
        };
        auto get0 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp, o, f);
        };
        auto cget1 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc, o, f) - center1;
        };
        auto get1 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc, o, f);
        };

        op(out[0],
              wx000 * center0
            + wx100 * (cget0(x0, fx0) + cget0(x1, fx1))
            + wx010 * (cget0(y0, fy0) + cget0(y1, fy1))
            + w2 * (
                  get1(x1+y0, fx1*fy0) + get1(x0+y1, fx0*fy1)
                - get1(x0+y0, fx0*fy0) - get1(x1+y1, fx1*fy1)
            )
        );

        op(out[osc],
              wy000 * center1
            + wy100 * (cget1(x0, fx0) + cget1(x1, fx1))
            + wy010 * (cget1(y0, fy0) + cget1(y1, fy1))
            + w2 * (
                  get0(x1+y0, fx1*fy0) + get0(x0+y1, fx0*fy1)
                - get0(x0+y0, fx0*fy0) - get0(x1+y1, fx1*fy1)
            )
        );
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
     kernel_lame(
        scalar_t * out, const offset_t sc[2], const offset_t stride[2],
        const reduce_t kernel[7])
    {
        offset_t sc0 = sc[0], sc1 = sc[1];
        const offset_t sx = stride[0], sy = stride[1];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0], w100 = kernel[1], w010 = kernel[2];
            op(out[0],      w000);
            op(out[-sx],    w100);
            op(out[+sx],    w100);
            op(out[-sy],    w010);
            op(out[+sy],    w010);
        };

        auto xxout = out, yyout = out + sc0 + sc1;
        setkernel(xxout, kernel);
        setkernel(yyout, kernel + 3);

        auto w2 = kernel[6];
        auto xyout = out + sc0, yxout = out + sc1;
        op(xyout[+sx+sy], -w2); op(xyout[-sx-sy], -w2);
        op(yxout[+sx+sy], -w2); op(yxout[-sx-sy], -w2);
        op(xyout[-sx+sy], +w2); op(xyout[+sx-sy], +w2);
        op(yxout[-sx+sy], +w2); op(yxout[+sx-sy], +w2);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline  void
    diag_lame(
        scalar_t * out, offset_t osc,
        const offset_t loc[2], const offset_t size[2],
        const reduce_t kernel[7])
    {
        offset_t  x = loc[0],     y = loc[1];
        offset_t nx = size[0],   ny = size[1];

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        signed char fy0 = bound_utils_y::sign(y-1, ny);
        signed char fy1 = bound_utils_y::sign(y+1, ny);

        reduce_t w2 = kernel[6];

        auto setdiag = [&](scalar_t & out, const reduce_t * kernel) {
            reduce_t w000 = kernel[0], w100 = kernel[1], w010 = kernel[2];
            w000 -= w100 * (fx0+fx1) + w010 * (fy0+fy1);
            op(out, w000);
        };

        setdiag(out[0],     kernel);
        setdiag(out[osc],   kernel + 3);
    }

    //------------------------------------------------------------------
    //                         ABSOLUTE JRLS
    //------------------------------------------------------------------

    // --- vel2mom ---

    template <OpType op = set>
    static inline __device__
    void vel2mom_absolute_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t osc, offset_t isc, const reduce_t kernel[2])
    {
        reduce_t w = static_cast<reduce_t>(*wgt);
        op(out[0],     kernel[0] * w * static_cast<reduce_t>(inp[0]));
        op(out[osc],   kernel[1] * w * static_cast<reduce_t>(inp[isc]));
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_absolute_jrls(
        scalar_t * out, const scalar_t * wgt,
        offset_t osc, const reduce_t kernel[2])
    {
        reduce_t w = static_cast<reduce_t>(*wgt);
        op(out[0],     kernel[0] * w);
        op(out[osc],   kernel[1] * w);
    }

    //------------------------------------------------------------------
    //                         MEMBRANE JRLS
    //------------------------------------------------------------------

    static const int kernelsize_membrane_jrls = kernelsize_membrane;

    __device__ static inline void
    make_kernel_membrane_jrls(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[2])
    {
        make_kernel_membrane(kernel, absolute, membrane, voxel_size);
        for (int k=0; k<kernelsize_membrane_jrls; ++k)
            kernel[k] *= 0.5;
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_membrane_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[2], const offset_t size[2],
        const offset_t istride[2], const offset_t wstride[2],
        offset_t osc, offset_t isc, const reduce_t kernel[6])
    {
        offset_t   x = loc[0],       y = loc[1];
        offset_t  nx = size[0],     ny = size[1];
        offset_t isx = istride[0], isy = istride[1];
        offset_t wsx = wstride[0], wsy = wstride[1];

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        signed char fy0 = bound_utils_y::sign(y-1, ny);
        signed char fy1 = bound_utils_y::sign(y+1, ny);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x);
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x);
        offset_t    iy0 = (bound_utils_y::index(y-1, ny) - y);
        offset_t    iy1 = (bound_utils_y::index(y+1, ny) - y);
        offset_t    wx0 = ix0 * wsx;
        offset_t    wx1 = ix1 * wsx;
        offset_t    wy0 = iy0 * wsy;
        offset_t    wy1 = iy1 * wsy;
        ix0 *= isx;
        ix1 *= isx;
        iy0 *= isy;
        iy1 *= isy;

        // --- load weight map ---

        reduce_t w111 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o) + w111;
        };
        reduce_t w011 = wget(wx0);
        reduce_t w211 = wget(wx1);
        reduce_t w101 = wget(wy0);
        reduce_t w121 = wget(wy1);

        // --- convolution ---

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t m000 = kernel[0], m100 = kernel[1], m010 = kernel[2];

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
               (m000*w111*2)*center
               + (m100*w011)*get(ix0, fx0) + (m100*w211)*get(ix1, fx1)
               + (m010*w101)*get(iy0, fy0) + (m010*w121)*get(iy1, fy1)
            );
        };

        conv(out,         inp,         kernel);
        conv(out + osc,   inp + isc,   kernel + 3);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_membrane_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[2], const offset_t size[2],
        const offset_t wstride[2], offset_t osc,
        const reduce_t kernel[6])
    {
        offset_t   x = loc[0],       y = loc[1];
        offset_t  nx = size[0],     ny = size[1];
        offset_t wsx = wstride[0], wsy = wstride[1];

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        signed char fy0 = bound_utils_y::sign(y-1, ny);
        signed char fy1 = bound_utils_y::sign(y+1, ny);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x) * wsx;
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x) * wsx;
        offset_t    iy0 = (bound_utils_y::index(y-1, ny) - y) * wsy;
        offset_t    iy1 = (bound_utils_y::index(y+1, ny) - y) * wsy;

        // --- load weight map ---

        reduce_t w111 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o) + w111;
        };
        reduce_t w011 = wget(ix0) * fx0;
        reduce_t w211 = wget(ix1) * fx1;
        reduce_t w101 = wget(iy0) * fy0;
        reduce_t w121 = wget(iy1) * fy1;

        // --- convolution ---

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t m000 = kernel[0], m100 = kernel[1], m010 = kernel[2];
            op(*out,
               m000*w111*2
               - m100*(w011 + w211)
               - m010*(w101 + w121)
            );
        };

        conv(out,         kernel);
        conv(out + osc,   kernel + 3);
    }

    //------------------------------------------------------------------
    //                         BENDING JRLS
    //------------------------------------------------------------------
#if 0

    static const int kernelsize_bending_jrls = 8;

    /* kernel = [
     *      lx, ly,
     *      k000, k100, k010, k200, k020, k110]
     *
     *      k000 = absolute
     *      k100 = -4 * lx * (lx + ly) * bending - lx * membrane
     *      k010 = -4 * ly * (lx + ly) * bending - ly * membrane
     *      k200 = lx * lx * bending
     *      k020 = ly * ly * bending
     *      k110 = 2 * lx * ly * bending
     *
     * where lx = 1/(vx[0]*vx[0])
     *       ly = 1/(vx[1]*vx[1])
     */
    static inline __device__ void
    make_kernel_bending_jrls(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);
        reduce_t vxy = vx + vy;

        kernel[0] = vx;
        kernel[1] = vy;

        kernel[2] = absolute;
        kernel[3] = (-4 * vxy * bending - membrane) * vx;
        kernel[4] = (-4 * vxy * bending - membrane) * vy;
        kernel[5] = vx * vx * bending;
        kernel[6] = vy * vy * bending;
        kernel[7] = 2 * vx * vy * bending;

        for (int k=4; k<kernelsize_bending_jrls; ++k)
            kernel[k] *= 0.25;
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_bending_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[2], const offset_t size[2],
        const offset_t istride[2], const offset_t wstride[2],
        offset_t osc, offset_t isc, const reduce_t kernel[8])
    {
        offset_t   x = loc[0],       y = loc[1];
        offset_t  nx = size[0],     ny = size[1];
        offset_t isx = istride[0], isy = istride[1];
        offset_t wsx = wstride[0], wsy = wstride[1];

        signed char fx0 = bound_utils_x::sign(x-2, nx);
        signed char fx1 = bound_utils_x::sign(x-1, nx);
        signed char fx3 = bound_utils_x::sign(x+1, nx);
        signed char fx4 = bound_utils_x::sign(x+2, nx);
        signed char fy0 = bound_utils_y::sign(y-2, ny);
        signed char fy1 = bound_utils_y::sign(y-1, ny);
        signed char fy3 = bound_utils_y::sign(y+1, ny);
        signed char fy4 = bound_utils_y::sign(y+2, ny);
        offset_t    ix0 = (bound_utils_x::index(x-2, nx) - x);
        offset_t    ix1 = (bound_utils_x::index(x-1, nx) - x);
        offset_t    ix3 = (bound_utils_x::index(x+1, nx) - x);
        offset_t    ix4 = (bound_utils_x::index(x+2, nx) - x);
        offset_t    iy0 = (bound_utils_y::index(y-2, ny) - y);
        offset_t    iy1 = (bound_utils_y::index(y-1, ny) - y);
        offset_t    iy3 = (bound_utils_y::index(y+1, ny) - y);
        offset_t    iy4 = (bound_utils_y::index(y+2, ny) - y);
        offset_t    wx0 = ix0 * wsx;
        offset_t    wx1 = ix1 * wsx;
        offset_t    wx3 = ix3 * wsx;
        offset_t    wx4 = ix4 * wsx;
        offset_t    wy0 = iy0 * wsy;
        offset_t    wy1 = iy1 * wsy;
        offset_t    wy3 = iy3 * wsy;
        offset_t    wy4 = iy4 * wsy;
        ix0 *= isx;
        ix1 *= isx;
        ix3 *= isx;
        ix4 *= isx;
        iy0 *= isy;
        iy1 *= isy;
        iy3 *= isy;
        iy4 *= isy;

        reduce_t vx = kernel[0], vy = kernel[1];
        kernel += 2;
        reduce_t b000 = kernel[0],
                 b100 = kernel[1], b010 = kernel[2],
                 b200 = kernel[3], b020 = kernel[4],
                 b110 = kernel[5];

        reduce_t w222 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o);
        };

        reduce_t w122 = wget(wx1);
        reduce_t w322 = wget(wx3);
        reduce_t w212 = wget(wy1);
        reduce_t w232 = wget(wy3);

        reduce_t w022 = wget(wx0);
        reduce_t w422 = wget(wx4);
        reduce_t w202 = wget(wy0);
        reduce_t w242 = wget(wy4);

        reduce_t w112 = wget(wx1+wy1);
        reduce_t w132 = wget(wx1+wy3);
        reduce_t w312 = wget(wx3+wy1);
        reduce_t w332 = wget(wx3+wy3);

        reduce_t m122 = (b100 - 2*b200) * (w222 + w122)
                        - 2*b200 * (w322 + w022)
                        - b110 * (w212 + w112 + w232 + w132);
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

        reduce_t center[] = {
            static_cast<reduce_t>(inp[0]),
            static_cast<reduce_t>(inp[isc]),
            static_cast<reduce_t>(inp[isc*2])
        };
        auto get = [&](offset_t d, offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp, o + d*isc, f) - center[d];
        };

        op(out, (b000*center[0] +
            (m122*get(0, ix1, fx1) +  m322*get(0, ix3, fx3) +
             m212*get(0, iy1, fy1) +  m232*get(0, iy3, fy3) +
             m221*get(0, iz1, fz1) +  m223*get(0, iz3, fz3)) +
            (m022*get(0, ix0, fx0) +  m422*get(0, ix4, fx4) +
             m202*get(0, iy0, fy0) +  m242*get(0, iy4, fy4) +
             m220*get(0, iz0, fz0) +  m224*get(0, iz4, fz4)) +
            (m112*get(0, ix1+iy1, fx1*fy1) +  m132*get(0, ix1+iy3, fx1*fy3) +
             m312*get(0, ix3+iy1, fx3*fy1) +  m332*get(0, ix3+iy3, fx3*fy3) +
             m121*get(0, ix1+iz1, fx1*fz1) +  m123*get(0, ix1+iz3, fx1*fz3) +
             m321*get(0, ix3+iz1, fx3*fz1) +  m323*get(0, ix3+iz3, fx3*fz3) +
             m211*get(0, iy1+iz1, fy1*fz1) +  m213*get(0, iy1+iz3, fy1*fz3) +
             m231*get(0, iy3+iz1, fy3*fz1) +  m233*get(0, iy3+iz3, fy3*fz3)))
             / vx);

        out += osc;

        op(out, (b000*center[1] +
            (m122*get(1, ix1, fx1) +  m322*get(1, ix3, fx3) +
             m212*get(1, iy1, fy1) +  m232*get(1, iy3, fy3) +
             m221*get(1, iz1, fz1) +  m223*get(1, iz3, fz3)) +
            (m022*get(1, ix0, fx0) +  m422*get(1, ix4, fx4) +
             m202*get(1, iy0, fy0) +  m242*get(1, iy4, fy4) +
             m220*get(1, iz0, fz0) +  m224*get(1, iz4, fz4)) +
            (m112*get(1, ix1+iy1, fx1*fy1) +  m132*get(1, ix1+iy3, fx1*fy3) +
             m312*get(1, ix3+iy1, fx3*fy1) +  m332*get(1, ix3+iy3, fx3*fy3) +
             m121*get(1, ix1+iz1, fx1*fz1) +  m123*get(1, ix1+iz3, fx1*fz3) +
             m321*get(1, ix3+iz1, fx3*fz1) +  m323*get(1, ix3+iz3, fx3*fz3) +
             m211*get(1, iy1+iz1, fy1*fz1) +  m213*get(1, iy1+iz3, fy1*fz3) +
             m231*get(1, iy3+iz1, fy3*fz1) +  m233*get(1, iy3+iz3, fy3*fz3)))
             / vy);

        out += osc;

        op(out, (b000*center[2] +
            (m122*get(2, ix1, fx1) +  m322*get(2, ix3, fx3) +
             m212*get(2, iy1, fy1) +  m232*get(2, iy3, fy3) +
             m221*get(2, iz1, fz1) +  m223*get(2, iz3, fz3)) +
            (m022*get(2, ix0, fx0) +  m422*get(2, ix4, fx4) +
             m202*get(2, iy0, fy0) +  m242*get(2, iy4, fy4) +
             m220*get(2, iz0, fz0) +  m224*get(2, iz4, fz4)) +
            (m112*get(2, ix1+iy1, fx1*fy1) +  m132*get(2, ix1+iy3, fx1*fy3) +
             m312*get(2, ix3+iy1, fx3*fy1) +  m332*get(2, ix3+iy3, fx3*fy3) +
             m121*get(2, ix1+iz1, fx1*fz1) +  m123*get(2, ix1+iz3, fx1*fz3) +
             m321*get(2, ix3+iz1, fx3*fz1) +  m323*get(2, ix3+iz3, fx3*fz3) +
             m211*get(2, iy1+iz1, fy1*fz1) +  m213*get(2, iy1+iz3, fy1*fz3) +
             m231*get(2, iy3+iz1, fy3*fz1) +  m233*get(2, iy3+iz3, fy3*fz3)))
             / vz);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_bending_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[3], const offset_t size[3],
        const offset_t wstride[3], offset_t osc, const reduce_t kernel[13])
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

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

        reduce_t vx = kernel[0], vy = kernel[1], vz = kernel[2];
        kernel += 3;
        reduce_t b000 = kernel[0],
                 b100 = kernel[1], b010 = kernel[2], b001 = kernel[3],
                 b200 = kernel[4], b020 = kernel[5], b002 = kernel[6],
                 b110 = kernel[7], b101 = kernel[8], b011 = kernel[9];

        reduce_t w222 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o);
        };

        reduce_t w122 = wget(ix1);
        reduce_t w322 = wget(ix3);
        reduce_t w212 = wget(iy1);
        reduce_t w232 = wget(iy3);
        reduce_t w221 = wget(iz1);
        reduce_t w223 = wget(iz3);

        reduce_t w022 = wget(ix0);
        reduce_t w422 = wget(ix4);
        reduce_t w202 = wget(iy0);
        reduce_t w242 = wget(iy4);
        reduce_t w220 = wget(iz0);
        reduce_t w224 = wget(iz4);

        reduce_t w112 = wget(ix1+iy1);
        reduce_t w132 = wget(ix1+iy3);
        reduce_t w312 = wget(ix3+iy1);
        reduce_t w332 = wget(ix3+iy3);
        reduce_t w121 = wget(ix1+iz1);
        reduce_t w123 = wget(ix1+iz3);
        reduce_t w321 = wget(ix3+iz1);
        reduce_t w323 = wget(ix3+iz3);
        reduce_t w211 = wget(iy1+iz1);
        reduce_t w213 = wget(iy1+iz3);
        reduce_t w231 = wget(iy3+iz1);
        reduce_t w233 = wget(iy3+iz3);

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

        b000 -= (m122*fx1 +  m322*fx3 +
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
                 m231*(fy3*fz1) +  m233*(fy3*fz3));

        op(out[0],     b000 / vx);
        op(out[osc],   b000 / vy);
        op(out[osc*2], b000 / vz);
    }
#endif

    //------------------------------------------------------------------
    //                           LAME JRLS
    //------------------------------------------------------------------

    static const int kernelsize_lame_jrls = 8;

    /* kernel = [wx000, wx100, wx010,
     *           wy000, wy100, wy010,
     *           d2, s2]
     *
     * wx100 = -(0.5*div + shears)
     * wx010 = -0.5*shears*(vy/vx)
     * wy100 = -0.5*shears*(vx/vy)
     * wy010 = -(0.5*div + shears)
     * d2    = 0.25*div
     * s2    = 0.25*shears
     */
    __device__ static inline void
    make_kernel_lame_jrls(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[2])
    {
        make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

        for (int k=0; k < 6; ++k)
            kernel[k] *= 0.5;

        kernel[6] = 0.25 * shears;
        kernel[7] = 0.25 * div;
    }

    // --- vel2mom ---

    template <OpType op = set>
    static inline __device__
    void vel2mom_lame_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[2], const offset_t size[2],
        const offset_t istride[2], const offset_t wstride[2],
        offset_t osc, offset_t isc, const reduce_t kernel[8])
    {
        offset_t   x = loc[0],       y = loc[1];
        offset_t  nx = size[0],     ny = size[1];
        offset_t isx = istride[0], isy = istride[1];
        offset_t wsx = wstride[0], wsy = wstride[1];

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        signed char fy0 = bound_utils_y::sign(y-1, ny);
        signed char fy1 = bound_utils_y::sign(y+1, ny);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x);
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x);
        offset_t    iy0 = (bound_utils_y::index(y-1, ny) - y);
        offset_t    iy1 = (bound_utils_y::index(y+1, ny) - y);
        offset_t    wx0 = ix0 * wsx;
        offset_t    wx1 = ix1 * wsx;
        offset_t    wy0 = iy0 * wsy;
        offset_t    wy1 = iy1 * wsy;
        ix0 *= isx;
        ix1 *= isx;
        iy0 *= isy;
        iy1 *= isy;

        // --- load weight map ---

        reduce_t w111 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o);
        };

        reduce_t w011 = wget(wx0);
        reduce_t w211 = wget(wx1);
        reduce_t w101 = wget(wy0);
        reduce_t w121 = wget(wy1);

        // --- weight map kernel

        reduce_t wx000 = kernel[0],  wx100 = kernel[1],  wx010 = kernel[2],
                 wy000 = kernel[3],  wy100 = kernel[4],  wy010 = kernel[5],
                 d2    = kernel[6],  s2    = kernel[7];

        // --- compute convolution ---

        reduce_t center0 = static_cast<reduce_t>(*inp);
        auto cget0 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp, o, f) - center0;
        };
        auto get0 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp, o, f);
        };

        reduce_t center1 = static_cast<reduce_t>(inp[isc]);
        auto cget1 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc, o, f) - center1;
        };
        auto get1 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc, o, f);
        };

        op(out[0],
           (wx000*w111*2)*center0
           + (wx100*(w011+w111))*cget0(ix0, fx0) + (wx100*(w211+w111))*cget0(ix1, fx1)
           + (wx010*(w101+w111))*cget0(iy0, fy0) + (wx010*(w121+w111))*cget0(iy1, fy1)
           - (d2*w011 + s2*w101)*get1(ix0+iy0, fx0*fy0)
           - (d2*w211 + s2*w121)*get1(ix1+iy1, fx1*fy1)
           + (d2*w011 + s2*w121)*get1(ix0+iy1, fx0*fy1)
           + (d2*w211 + s2*w101)*get1(ix1+iy0, fx1*fy0)
        );

        op(out[osc],
           (wy000*w111*2)*center1
           + (wy100*(w011+w111))*cget1(ix0, fx0) + (wy100*(w211+w111))*cget1(ix1, fx1)
           + (wy010*(w101+w111))*cget1(iy0, fy0) + (wy010*(w121+w111))*cget1(iy1, fy1)
           - (d2*w101 + s2*w011)*get0(iy0+ix0, fy0*fx0)
           - (d2*w121 + s2*w211)*get0(iy1+ix1, fy1*fx1)
           + (d2*w101 + s2*w211)*get0(iy0+ix1, fy0*fx1)
           + (d2*w121 + s2*w011)*get0(iy1+ix0, fy1*fx0)
        );

    }

    // --- diagonal ---

    template <OpType op = set>
    static inline __device__
    void diag_lame_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[2], const offset_t size[2],
        const offset_t wstride[2], offset_t osc, const reduce_t kernel[8])
    {
        offset_t   x = loc[0],       y = loc[1];
        offset_t  nx = size[0],     ny = size[1];
        offset_t wsx = wstride[0], wsy = wstride[1];

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        signed char fy0 = bound_utils_y::sign(y-1, ny);
        signed char fy1 = bound_utils_y::sign(y+1, ny);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x) * wsx;
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x) * wsx;
        offset_t    iy0 = (bound_utils_y::index(y-1, ny) - y) * wsy;
        offset_t    iy1 = (bound_utils_y::index(y+1, ny) - y) * wsy;

        // --- load weight map ---

        reduce_t w0 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o) + w0;
        };

        reduce_t wx = wget(ix0) * fx0 + wget(ix1) * fx1;
        reduce_t wy = wget(iy0) * fy0 + wget(iy1) * fy1;

        // --- compute convolution ---

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t m000 = kernel[0], m100 = kernel[1], m010 = kernel[2];
            op(*out, m000*w0*2 - m100*wx - m010*wy);
        };

        conv(out,         kernel);
        conv(out + osc,   kernel + 3);
    }

};

} // namespace reg_grid
} // namespace jf

#endif // JF_REGULARISERS_GRID_2D
