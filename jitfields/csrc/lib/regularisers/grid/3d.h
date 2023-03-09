#ifndef JF_REGULARISERS_GRID_3D
#define JF_REGULARISERS_GRID_3D
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
          bound::type BX, bound::type BY, bound::type BZ>
struct RegGrid<three, scalar_t, reduce_t, offset_t, BX, BY, BZ> {
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    using bound_utils_z = bound::utils<BZ>;
    typedef scalar_t & (*OpType)(scalar_t &, const reduce_t &);

    //------------------------------------------------------------------
    //                            ABSOLUTE
    //------------------------------------------------------------------

    static const int kernelsize_absolute = 3;

    /// kernel <- [absx, absy, absz]
    __device__ static inline void
    make_kernel_absolute(
        reduce_t * kernel, reduce_t absolute, const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);
        kernel[0] = absolute / vx;
        kernel[1] = absolute / vy;
        kernel[2] = absolute / vz;
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_absolute(
        scalar_t * out, const scalar_t * inp,
        offset_t osc, offset_t isc, const reduce_t kernel[3])
    {
        op(out[0],     kernel[0] * inp[0]);
        op(out[osc],   kernel[1] * inp[isc]);
        op(out[osc*2], kernel[2] * inp[isc*2]);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline  void
    kernel_absolute(scalar_t * out, offset_t osc, const reduce_t kernel[3])
    {
        op(out[0],     kernel[0]);
        op(out[osc],   kernel[1]);
        op(out[osc*2], kernel[2]);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline  void
    diag_absolute(scalar_t * out, offset_t osc, const reduce_t kernel[3])
    {
        return kernel_absolute(out, osc, kernel);
    }

    //------------------------------------------------------------------
    //                            MEMBRANE
    //------------------------------------------------------------------

    static const int kernelsize_membrane = 12;

    /// kernel <- [absx, wx100, wx010, wx001,
    ///            absy, wy100, wy010, wy001,
    ///            absz, wz100, wz010, wz001]
    __device__ static inline void
    make_kernel_membrane(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);
        kernel[0]  = absolute / vx;
        kernel[1]  = -membrane;
        kernel[2]  = -membrane * (vy/vx);
        kernel[3]  = -membrane * (vz/vx);
        kernel[4]  = absolute / vy;
        kernel[5]  = -membrane * (vx/vy);
        kernel[6]  = -membrane;
        kernel[7]  = -membrane * (vz/vy);
        kernel[8]  = absolute / vz;
        kernel[9]  = -membrane * (vx/vz);
        kernel[10] = -membrane * (vy/vz);
        kernel[11] = -membrane;
    }

    /// kernel <- [wx000, wx100, wx010, wx001,
    ///            wy000, wy100, wy010, wy001,
    ///            wz000, wz100, wz010, wz001]
    __device__ static inline void
    make_fullkernel_membrane(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);
        kernel[0]  = (absolute + 2 * membrane * (vx + vy + vz)) / vx;
        kernel[1]  = -membrane;
        kernel[2]  = -membrane * (vy/vx);
        kernel[3]  = -membrane * (vz/vx);
        kernel[4]  = (absolute + 2 * membrane * (vx + vy + vz)) / vy;
        kernel[5]  = -membrane * (vx/vy);
        kernel[6]  = -membrane;
        kernel[7]  = -membrane * (vz/vy);
        kernel[8]  = (absolute + 2 * membrane * (vx + vy + vz)) / vz;
        kernel[9]  = -membrane * (vx/vz);
        kernel[10] = -membrane * (vy/vz);
        kernel[11] = -membrane;
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_membrane(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[3], const offset_t size[3], const offset_t stride[3],
        offset_t osc, offset_t isc, const reduce_t kernel[12])
    {
        offset_t  x = loc[0],     y = loc[1],     z = loc[2];
        offset_t nx = size[0],   ny = size[1],   nz = size[2];
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];

        offset_t x0 = x-1, x1 = x+1, y0 = y-1, y1 = y+1, z0 = z-1, z1 = z+1;
        signed char fx0 = bound_utils_x::sign(x0, nx);
        signed char fx1 = bound_utils_x::sign(x1, nx);
        signed char fy0 = bound_utils_y::sign(y0, ny);
        signed char fy1 = bound_utils_y::sign(y1, ny);
        signed char fz0 = bound_utils_z::sign(z0, nz);
        signed char fz1 = bound_utils_z::sign(z1, nz);
        x0 = (bound_utils_x::index(x0, nx) - x) * sx;
        x1 = (bound_utils_x::index(x1, nx) - x) * sx;
        y0 = (bound_utils_y::index(y0, ny) - y) * sy;
        y1 = (bound_utils_y::index(y1, ny) - y) * sy;
        z0 = (bound_utils_z::index(z0, nz) - z) * sz;
        z1 = (bound_utils_z::index(z1, nz) - z) * sz;

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t center = static_cast<reduce_t>(inp[0]);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out, kernel[0] * center +
                     kernel[1] * (get(x0, fx0) + get(x1, fx1)) +
                     kernel[2] * (get(y0, fy0) + get(y1, fy1)) +
                     kernel[3] * (get(z0, fz0) + get(z1, fz1)));
        };

        conv(out,         inp,         kernel);
        conv(out + osc,   inp + isc,   kernel + 4);
        conv(out + osc*2, inp + isc*2, kernel + 8);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
    kernel_membrane(
        scalar_t * out, offset_t sc, const offset_t stride[3],
        const reduce_t kernel[12])
    {
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0], w100 = kernel[1],
                     w010 = kernel[2], w001 = kernel[3];
            op(out[0],   w000);
            op(out[-sx], w100);
            op(out[+sx], w100);
            op(out[-sy], w010);
            op(out[+sy], w010);
            op(out[-sz], w001);
            op(out[+sz], w001);
        };

        setkernel(out,        kernel);
        setkernel(out + sc,   kernel + 4);
        setkernel(out + sc*2, kernel + 8);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_membrane(
        scalar_t * out, offset_t osc,
         const offset_t loc[3], const offset_t size[3],
         const reduce_t kernel[12])
    {
        offset_t  x = loc[0],   y = loc[1],   z = loc[2];
        offset_t nx = size[0], ny = size[1], nz = size[2];

        signed char fx = bound_utils_x::sign(x-1, nx)
                       + bound_utils_x::sign(x+1, nx);
        signed char fy = bound_utils_y::sign(y-1, ny)
                       + bound_utils_y::sign(y+1, ny);
        signed char fz = bound_utils_z::sign(z-1, nz)
                       + bound_utils_z::sign(z+1, nz);

         op(out[0],     kernel[0] - kernel[1]*fx - kernel[2]*fy  - kernel[3]*fz);
         op(out[osc],   kernel[4] - kernel[5]*fx - kernel[6]*fy  - kernel[7]*fz);
         op(out[osc*2], kernel[8] - kernel[9]*fx - kernel[10]*fy - kernel[11]*fz);
    }

    //------------------------------------------------------------------
    //                            BENDING
    //------------------------------------------------------------------

    static const int kernelsize_bending = 30;

    /// kernel <- [
    ///     absx, wx100, wx010, wx001, wx200, wx020, wx002, wx110, wx101, wx011,
    ///     absy, wy100, wy010, wy001, wy200, wy020, wy002, wy110, wy101, wy011,
    ///     absz, wz100, wz010, wz001, wz200, wz020, wz002, wz110, wz101, wz011]
    __device__ static inline void
    make_kernel_bending(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);

        reduce_t w100 = -4 * bending * vx * (vx + vy + vz) - membrane * vx;
        reduce_t w010 = -4 * bending * vy * (vx + vy + vz) - membrane * vy;
        reduce_t w001 = -4 * bending * vz * (vx + vy + vz) - membrane * vz;
        reduce_t w200 = bending * vx * vx;
        reduce_t w020 = bending * vy * vy;
        reduce_t w002 = bending * vz * vz;
        reduce_t w110 = 2 * bending * vx * vy;
        reduce_t w101 = 2 * bending * vx * vz;
        reduce_t w011 = 2 * bending * vy * vz;
        reduce_t w000 = absolute;

        kernel[0]  = w000 / vx;
        kernel[1]  = w100 / vx;
        kernel[2]  = w010 / vx;
        kernel[3]  = w001 / vx;
        kernel[4]  = w200 / vx;
        kernel[5]  = w020 / vx;
        kernel[6]  = w002 / vx;
        kernel[7]  = w110 / vx;
        kernel[8]  = w101 / vx;
        kernel[9]  = w011 / vx;

        kernel[10] = w000 / vy;
        kernel[11] = w100 / vy;
        kernel[12] = w010 / vy;
        kernel[13] = w001 / vy;
        kernel[14] = w200 / vy;
        kernel[15] = w020 / vy;
        kernel[16] = w002 / vy;
        kernel[17] = w110 / vy;
        kernel[18] = w101 / vy;
        kernel[19] = w011 / vy;

        kernel[20] = w000 / vz;
        kernel[21] = w100 / vz;
        kernel[22] = w010 / vz;
        kernel[23] = w001 / vz;
        kernel[24] = w200 / vz;
        kernel[25] = w020 / vz;
        kernel[26] = w002 / vz;
        kernel[27] = w110 / vz;
        kernel[28] = w101 / vz;
        kernel[29] = w011 / vz;
    }

    /// kernel <- [
    ///     wx000, wx100, wx010, wx001, wx200, wx020, wx002, wx110, wx101, wx011,
    ///     wy000, wy100, wy010, wy001, wy200, wy020, wy002, wy110, wy101, wy011,
    ///     wz000, wz100, wz010, wz001, wz200, wz020, wz002, wz110, wz101, wz011]
    __device__ static inline void
    make_fullkernel_bending(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);

        reduce_t w100 = -4 * bending * vx * (vx + vy + vz) - membrane * vx;
        reduce_t w010 = -4 * bending * vy * (vx + vy + vz) - membrane * vy;
        reduce_t w001 = -4 * bending * vz * (vx + vy + vz) - membrane * vz;
        reduce_t w200 = bending * vx * vx;
        reduce_t w020 = bending * vy * vy;
        reduce_t w002 = bending * vz * vz;
        reduce_t w110 = 2 * bending * vx * vy;
        reduce_t w101 = 2 * bending * vx * vz;
        reduce_t w011 = 2 * bending * vy * vz;
        reduce_t w000 = absolute
                      - 2 * (w100 + w010 + w001 + w200 + w020 + w002)
                      - 4 * (w110 + w101 + w011);

        kernel[0]  = w000 / vx;
        kernel[1]  = w100 / vx;
        kernel[2]  = w010 / vx;
        kernel[3]  = w001 / vx;
        kernel[4]  = w200 / vx;
        kernel[5]  = w020 / vx;
        kernel[6]  = w002 / vx;
        kernel[7]  = w110 / vx;
        kernel[8]  = w101 / vx;
        kernel[9]  = w011 / vx;

        kernel[10] = w000 / vy;
        kernel[11] = w100 / vy;
        kernel[12] = w010 / vy;
        kernel[13] = w001 / vy;
        kernel[14] = w200 / vy;
        kernel[15] = w020 / vy;
        kernel[16] = w002 / vy;
        kernel[17] = w110 / vy;
        kernel[18] = w101 / vy;
        kernel[19] = w011 / vy;

        kernel[20] = w000 / vz;
        kernel[21] = w100 / vz;
        kernel[22] = w010 / vz;
        kernel[23] = w001 / vz;
        kernel[24] = w200 / vz;
        kernel[25] = w020 / vz;
        kernel[26] = w002 / vz;
        kernel[27] = w110 / vz;
        kernel[28] = w101 / vz;
        kernel[29] = w011 / vz;
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_bending(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[3], const offset_t size[3],
        const offset_t stride[3], offset_t osc, offset_t isc,
        const reduce_t kernel[30])
    {
        offset_t  x = loc[0],     y = loc[1],     z = loc[2];
        offset_t nx = size[0],   ny = size[1],   nz = size[2];
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];

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

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1], w010 = kernel[2], w001 = kernel[3],
                     w200 = kernel[4], w020 = kernel[5], w002 = kernel[6],
                     w110 = kernel[7], w101 = kernel[8], w011 = kernel[9];

            reduce_t center = static_cast<reduce_t>(inp[0]);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
                  w000 * center
                + w100 * (get(x0, fx0) + get(x1, fx1))
                + w010 * (get(y0, fy0) + get(y1, fy1))
                + w001 * (get(z0, fz0) + get(z1, fz1))
                + w200 * (get(x00, fx00) + get(x11, fx11))
                + w020 * (get(y00, fy00) + get(y11, fy11))
                + w002 * (get(z00, fz00) + get(z11, fz11))
                + w110 * (get(x0+y0, fx0*fy0) + get(x1+y0, fx1*fy0) +
                          get(x0+y1, fx0*fy1) + get(x1+y1, fx1*fy1))
                + w101 * (get(x0+z0, fx0*fz0) + get(x1+z0, fx1*fz0) +
                          get(x0+z1, fx0*fz1) + get(x1+z1, fx1*fz1))
                + w011 * (get(y0+z0, fy0*fz0) + get(y1+z0, fy1*fz0) +
                          get(y0+z1, fy0*fz1) + get(y1+z1, fy1*fz1))
            );
        };

        conv(out,         inp,         kernel);
        conv(out + osc,   inp + isc,   kernel + 10);
        conv(out + osc*2, inp + isc*2, kernel + 20);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
     kernel_bending(
        scalar_t * out, offset_t sc, const offset_t stride[3],
        const reduce_t kernel[30])
    {
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];

        auto setkernel = [&](scalar_t * o, const reduce_t * ker) {
            reduce_t w000 = ker[0],
                     w100 = ker[1], w010 = ker[2], w001 = ker[3],
                     w200 = ker[4], w020 = ker[5], w002 = ker[6],
                     w110 = ker[7], w101 = ker[8], w011 = ker[9];
            op(o[0],      w000);
            op(o[-sx],    w100);
            op(o[+sx],    w100);
            op(o[-sy],    w010);
            op(o[+sy],    w010);
            op(o[-sz],    w001);
            op(o[+sz],    w001);
            op(o[-sx*2],  w200);
            op(o[+sx*2],  w200);
            op(o[-sy*2],  w020);
            op(o[+sy*2],  w020);
            op(o[-sz*2],  w002);
            op(o[+sz*2],  w002);
            op(o[-sx-sy], w110);
            op(o[-sx+sy], w110);
            op(o[+sx-sy], w110);
            op(o[+sx+sy], w110);
            op(o[-sx-sz], w101);
            op(o[-sx+sz], w101);
            op(o[+sx-sz], w101);
            op(o[+sx+sz], w101);
            op(o[-sy-sz], w011);
            op(o[-sy+sz], w011);
            op(o[+sy-sz], w011);
            op(o[+sy+sz], w011);
        };

        setkernel(out,        kernel);
        setkernel(out + sc,   kernel + 10);
        setkernel(out + sc*2, kernel + 20);
    }

    // --- diagonal ---

    template <OpType op = set>
    static inline __device__ void
    diag_bending(
        scalar_t * out, offset_t osc,
        const offset_t loc[3], const offset_t size[3],
        const reduce_t kernel[30])
    {
        /* NOTE:
         * kernel = [
         *      absx, wx100, wx010, wx001, wx200, wx020, wx002, wx110, wx101, wx011,
         *      absy, wy100, wy010, wy001, wy200, wy020, wy002, wy110, wy101, wy011,
         *      absz, wz100, wz010, wz001, wz200, wz020, wz002, wz110, wz101, wz011]
         */

        offset_t  x = loc[0],     y = loc[1],     z = loc[2];
        offset_t nx = size[0],   ny = size[1],   nz = size[2];

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

        auto setdiag = [&](scalar_t & out, const reduce_t * kernel) {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1], w010 = kernel[2], w001 = kernel[3],
                     w200 = kernel[4], w020 = kernel[5], w002 = kernel[6],
                     w110 = kernel[7], w101 = kernel[8], w011 = kernel[9];
            w000 -=   w100 * (fx0 + fx1)   + w010 * (fy0 + fy1)   + w001 * (fz0 + fz1)
                    + w200 * (fx00 + fx11) + w020 * (fy00 + fy11) + w002 * (fz00 + fz11)
                    + w110 * (fx0*fy0 + fx1*fy0 + fx1*fy0 + fx1*fy1)
                    + w101 * (fx0*fz0 + fx1*fz0 + fx1*fz0 + fx1*fz1)
                    + w011 * (fy0*fz0 + fy1*fz0 + fy1*fz0 + fy1*fz1);
            op(out, w000);
        };
        setdiag(out[0],     kernel);
        setdiag(out[osc],   kernel + 10);
        setdiag(out[osc*2], kernel + 20);
    }

    //------------------------------------------------------------------
    //                          LAME + BENDING
    //------------------------------------------------------------------

    static const int kernelsize_all = 31;

    /// kernel <- [
    ///      absx, wx100, wx010, wx001, wx200, wx020, wx002, wx110, wx101, wx001,
    ///      absy, wy100, wy010, wy001, wy200, wy020, wy002, wy110, wy101, wy001,
    ///      absz, wz100, wz010, wz001, wz200, wz020, wz002, wz110, wz101, wz001,
    ///      ww]
    static inline __device__ void
    make_kernel_all(
        reduce_t * kernel,
        reduce_t absolute, reduce_t membrane, reduce_t bending,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);
        reduce_t vxyz = vx + vy + vz;

        reduce_t w100 = (-4 * bending * vxyz - membrane) * vx;
        reduce_t w010 = (-4 * bending * vxyz - membrane) * vy;
        reduce_t w001 = (-4 * bending * vxyz - membrane) * vz;
        reduce_t w200 = bending * vx * vx;
        reduce_t w020 = bending * vy * vy;
        reduce_t w002 = bending * vz * vz;
        reduce_t w110 = 2 * bending * vx * vy;
        reduce_t w101 = 2 * bending * vx * vz;
        reduce_t w011 = 2 * bending * vy * vz;
        reduce_t  w000 = absolute;

        kernel[0]  = w000 / vx;
        kernel[1]  = w100 / vx - 2*shears - div;
        kernel[2]  = w010 / vx - shears*(vy/vx);
        kernel[3]  = w001 / vx - shears*(vz/vx);
        kernel[4]  = w200 / vx;
        kernel[5]  = w020 / vx;
        kernel[6]  = w002 / vx;
        kernel[7]  = w110 / vx;
        kernel[8]  = w101 / vx;
        kernel[9]  = w011 / vx;

        kernel[10] = w000 / vy;
        kernel[11] = w100 / vy - shears*(vx/vy);
        kernel[12] = w010 / vy - 2*shears - div;
        kernel[13] = w001 / vy - shears*(vz/vy);
        kernel[14] = w200 / vy;
        kernel[15] = w020 / vy;
        kernel[16] = w002 / vy;
        kernel[17] = w110 / vy;
        kernel[18] = w101 / vy;
        kernel[19] = w011 / vy;

        kernel[20] = w000 / vz;
        kernel[21] = w100 / vz - shears*(vx/vz);
        kernel[22] = w010 / vz - shears*(vy/vz);
        kernel[23] = w001 / vz - 2*shears - div;
        kernel[24] = w200 / vz;
        kernel[25] = w020 / vz;
        kernel[26] = w002 / vz;
        kernel[27] = w110 / vz;
        kernel[28] = w101 / vz;
        kernel[29] = w011 / vz;

        kernel[30] = 0.25 * (shears + div);
    }

    /// kernel <- [
    ///      wx000, wx100, wx010, wx001, wx200, wx020, wx002, wx110, wx101, wx001,
    ///      wy000, wy100, wy010, wy001, wy200, wy020, wy002, wy110, wy101, wy001,
    ///      wz000, wz100, wz010, wz001, wz200, wz020, wz002, wz110, wz101, wz001,
    ///      ww]
    static inline __device__ void make_fullkernel_all(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        reduce_t shears, reduce_t div,
        const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);
        reduce_t vxyz = vx + vy + vz;

        reduce_t w100 = (-4 * bending * vxyz - membrane) * vx;
        reduce_t w010 = (-4 * bending * vxyz - membrane) * vy;
        reduce_t w001 = (-4 * bending * vxyz - membrane) * vz;
        reduce_t w200 = bending * vx * vx;
        reduce_t w020 = bending * vy * vy;
        reduce_t w002 = bending * vz * vz;
        reduce_t w110 = 2 * bending * vx * vy;
        reduce_t w101 = 2 * bending * vx * vz;
        reduce_t w011 = 2 * bending * vy * vz;
        reduce_t w000 = absolute
                      - 2 * (w100 + w010 + w001 + w200 + w020 + w002)
                      - 4 * (w110 + w101 + w011);

        kernel[0]  = w000 / vx + 2*shears*(2*vx+vy+vz)/vx + 2*div;
        kernel[1]  = w100 / vx - 2*shears - div;
        kernel[2]  = w010 / vx - shears*(vy/vx);
        kernel[3]  = w001 / vx - shears*(vz/vx);
        kernel[4]  = w200 / vx;
        kernel[5]  = w020 / vx;
        kernel[6]  = w002 / vx;
        kernel[7]  = w110 / vx;
        kernel[8]  = w101 / vx;
        kernel[9]  = w011 / vx;

        kernel[10] = w000 / vy + 2*shears*(vx+2*vy+vz)/vy + 2*div;
        kernel[11] = w100 / vy - shears*(vx/vy);
        kernel[12] = w010 / vy - 2*shears - div;
        kernel[13] = w001 / vy - shears*(vz/vy);
        kernel[14] = w200 / vy;
        kernel[15] = w020 / vy;
        kernel[16] = w002 / vy;
        kernel[17] = w110 / vy;
        kernel[18] = w101 / vy;
        kernel[19] = w011 / vy;

        kernel[20] = w000 / vz + 2*shears*(vx+vy+2*vz)/vz + 2*div;
        kernel[21] = w100 / vz - shears*(vx/vz);
        kernel[22] = w010 / vz - shears*(vy/vz);
        kernel[23] = w001 / vz - 2*shears - div;
        kernel[24] = w200 / vz;
        kernel[25] = w020 / vz;
        kernel[26] = w002 / vz;
        kernel[27] = w110 / vz;
        kernel[28] = w101 / vz;
        kernel[29] = w011 / vz;

        kernel[30] = 0.25 * (shears + div);
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_all(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[3], const offset_t size[3],
        const offset_t stride[3], offset_t osc, offset_t isc,
        const reduce_t kernel[31])
    {
        offset_t  x = loc[0],     y = loc[1],     z = loc[2];
        offset_t nx = size[0],   ny = size[1],   nz = size[2];
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];

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

        reduce_t center0 = static_cast<reduce_t>(inp[0]),
                 center1 = static_cast<reduce_t>(inp[isc]),
                 center2 = static_cast<reduce_t>(inp[isc*2]);

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
        auto cget2 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc*2, o, f) - center2;
        };
        auto get2 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc*2, o, f);
        };

        auto w2 = kernel[30];

        {
            reduce_t wx000 = kernel[0],
                     wx100 = kernel[1], wx010 = kernel[2], wx001 = kernel[3],
                     wx200 = kernel[4], wx020 = kernel[5], wx002 = kernel[6],
                     wx110 = kernel[7], wx101 = kernel[8], wx011 = kernel[9];

            op(out[0],
                  wx000 * center0
                + wx100 * (cget0(x0, fx0)        + cget0(x1, fx1))
                + wx010 * (cget0(y0, fy0)        + cget0(y1, fy1))
                + wx001 * (cget0(z0, fz0)        + cget0(z1, fz1))
                + wx200 * (cget0(x00, fx00)      + cget0(x11, fx11))
                + wx020 * (cget0(y00, fy00)      + cget0(y11, fy11))
                + wx002 * (cget0(z00, fz00)      + cget0(z11, fz11))
                + wx110 * (cget0(x0+y0, fx0*fy0) + cget0(x1+y0, fx1*fy0) +
                           cget0(x0+y1, fx0*fy1) + cget0(x1+y1, fx1*fy1))
                + wx101 * (cget0(x0+z0, fx0*fz0) + cget0(x1+z0, fx1*fz0) +
                           cget0(x0+z1, fx0*fz1) + cget0(x1+z1, fx1*fz1))
                + wx011 * (cget0(y0+z0, fy0*fz0) + cget0(y1+z0, fy1*fz0) +
                           cget0(y0+z1, fy0*fz1) + cget0(y1+z1, fy1*fz1))
                + w2 * (
                      get1(x1+y0, fx1*fy0) + get1(x0+y1, fx0*fy1)
                    - get1(x0+y0, fx1*fy1) - get1(x1+y1, fx1*fy1)
                    + get2(x1+z0, fx1*fz0) + get2(x0+z1, fx0*fz1)
                    - get2(x0+z0, fx1*fz1) - get2(x1+z1, fx1*fz1)
                )
            );
        }

        kernel += 10;

        {
            reduce_t wy000 = kernel[0],
                     wy100 = kernel[1], wy010 = kernel[2], wy001 = kernel[3],
                     wy200 = kernel[4], wy020 = kernel[5], wy002 = kernel[6],
                     wy110 = kernel[7], wy101 = kernel[8], wy011 = kernel[9];

            op(out[osc],
                  wy000 * center1
                + wy100 * (cget1(x0, fx0)        + cget1(x1, fx1))
                + wy010 * (cget1(y0, fy0)        + cget1(y1, fy1))
                + wy001 * (cget1(z0, fz0)        + cget1(z1, fz1))
                + wy200 * (cget1(x00, fx00)      + cget1(x11, fx11))
                + wy020 * (cget1(y00, fy00)      + cget1(y11, fy11))
                + wy002 * (cget1(z00, fz00)      + cget1(z11, fz11))
                + wy110 * (cget1(x0+y0, fx0*fy0) + cget1(x1+y0, fx1*fy0) +
                           cget1(x0+y1, fx0*fy1) + cget1(x1+y1, fx1*fy1))
                + wy101 * (cget1(x0+z0, fx0*fz0) + cget1(x1+z0, fx1*fz0) +
                           cget1(x0+z1, fx0*fz1) + cget1(x1+z1, fx1*fz1))
                + wy011 * (cget1(y0+z0, fy0*fz0) + cget1(y1+z0, fy1*fz0) +
                           cget1(y0+z1, fy0*fz1) + cget1(y1+z1, fy1*fz1))
                + w2 * (
                      get0(x1+y0, fx1*fy0) + get0(x0+y1, fx0*fy1)
                    - get0(x0+y0, fx1*fy1) - get0(x1+y1, fx1*fy1)
                    + get2(y1+z0, fy1*fz0) + get2(y0+z1, fy0*fz1)
                    - get2(y0+z0, fy1*fz1) - get2(y1+z1, fy1*fz1)
                )
            );
        }

        kernel += 10;

        {
            reduce_t wz000 = kernel[0],
                     wz100 = kernel[1], wz010 = kernel[2], wz001 = kernel[3],
                     wz200 = kernel[4], wz020 = kernel[5], wz002 = kernel[6],
                     wz110 = kernel[7], wz101 = kernel[8], wz011 = kernel[9];

            op(out[osc*2],
                  wz000 * center2
                + wz100 * (cget2(x0, fx0)        + cget2(x1, fx1))
                + wz010 * (cget2(y0, fy0)        + cget2(y1, fy1))
                + wz001 * (cget2(z0, fz0)        + cget2(z1, fz1))
                + wz200 * (cget2(x00, fx00)      + cget2(x11, fx11))
                + wz020 * (cget2(y00, fy00)      + cget2(y11, fy11))
                + wz002 * (cget2(z00, fz00)      + cget2(z11, fz11))
                + wz110 * (cget2(x0+y0, fx0*fy0) + cget2(x1+y0, fx1*fy0) +
                           cget2(x0+y1, fx0*fy1) + cget2(x1+y1, fx1*fy1))
                + wz101 * (cget2(x0+z0, fx0*fz0) + cget2(x1+z0, fx1*fz0) +
                           cget2(x0+z1, fx0*fz1) + cget2(x1+z1, fx1*fz1))
                + wz011 * (cget2(y0+z0, fy0*fz0) + cget2(y1+z0, fy1*fz0) +
                           cget2(y0+z1, fy0*fz1) + cget2(y1+z1, fy1*fz1))
                + w2 * (
                      get0(x1+z0, fx1*fz0) + get0(x0+z1, fx0*fz1)
                    - get0(x0+z0, fx1*fz1) - get0(x1+z1, fx1*fz1)
                    + get1(y1+z0, fy1*fz0) + get1(y0+z1, fy0*fz1)
                    - get1(y0+z0, fy1*fz1) - get1(y1+z1, fy1*fz1)
                )
            );
        }
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
    kernel_all(
        scalar_t * out, const offset_t sc[2],
        const offset_t stride[3], const reduce_t kernel[31])
    {
        reduce_t sc0 = sc[0], sc1 = sc[1];
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1], w010 = kernel[2], w001 = kernel[3],
                     w200 = kernel[4], w020 = kernel[5], w002 = kernel[6],
                     w110 = kernel[7], w101 = kernel[8], w011 = kernel[9];
            op(out[0],      w000);
            op(out[-sx],    w100);
            op(out[+sx],    w100);
            op(out[-sy],    w010);
            op(out[+sy],    w010);
            op(out[-sz],    w001);
            op(out[+sz],    w001);
            op(out[-sx*2],  w200);
            op(out[+sx*2],  w200);
            op(out[-sy*2],  w020);
            op(out[+sy*2],  w020);
            op(out[-sz*2],  w002);
            op(out[+sz*2],  w002);
            op(out[-sx-sy], w110);
            op(out[-sx+sy], w110);
            op(out[+sx-sy], w110);
            op(out[+sx+sy], w110);
            op(out[-sx-sz], w101);
            op(out[-sx+sz], w101);
            op(out[+sx-sz], w101);
            op(out[+sx+sz], w101);
            op(out[-sy-sz], w011);
            op(out[-sy+sz], w011);
            op(out[+sy-sz], w011);
            op(out[+sy+sz], w011);
        };

        auto xxout = out, yyout = out + sc0 + sc1, zzout = out + (sc0 + sc1)*2;
        setkernel(xxout, kernel);
        setkernel(yyout, kernel + 10);
        setkernel(zzout, kernel + 20);

        auto w2 = kernel[30];
        auto xyout = out + sc0,         yxout = out + sc1,
             xzout = out + sc0*2,       zxout = out + sc1*2,
             yzout = out + sc0*2 + sc1, zyout = out + sc0 + sc1*2;
        op(xyout[+sx+sy], -w2); op(xyout[-sx-sy], -w2); op(yxout[+sx+sy], -w2); op(yxout[-sx-sy], -w2);
        op(xzout[+sx+sz], -w2); op(xzout[-sx-sz], -w2); op(zxout[+sx+sz], -w2); op(zxout[-sx-sz], -w2);
        op(yzout[+sy+sz], -w2); op(yzout[-sy-sz], -w2); op(zyout[+sy+sz], -w2); op(zyout[-sy-sz], -w2);
        op(xyout[-sx+sy], +w2); op(xyout[+sx-sy], +w2); op(yxout[-sx+sy], +w2); op(yxout[+sx-sy], +w2);
        op(xzout[-sx+sz], +w2); op(xzout[+sx-sz], +w2); op(zxout[-sx+sz], +w2); op(zxout[+sx-sz], +w2);
        op(yzout[-sy+sz], +w2); op(yzout[+sy-sz], +w2); op(zyout[-sy+sz], +w2); op(zyout[+sy-sz], +w2);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline  void
    diag_all(
        scalar_t * out, offset_t osc,
        const offset_t loc[3], const offset_t size[3],
        const reduce_t kernel[31])
    {
        offset_t  x = loc[0],   y = loc[1],   z = loc[2];
        offset_t nx = size[0], ny = size[1], nz = size[2];

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

        reduce_t w2 = kernel[30];

        auto setdiag = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1], w010 = kernel[2], w001 = kernel[3],
                     w200 = kernel[4], w020 = kernel[5], w002 = kernel[6],
                     w110 = kernel[7], w101 = kernel[8], w011 = kernel[9];
            w000 -=   w100 * (fx0 + fx1)   + w010 * (fy0 + fy1)   + w001 * (fz0 + fz1)
                    + w200 * (fx00 + fx11) + w020 * (fy00 + fy11) + w002 * (fz00 + fz11)
                    + w110 * (fx0*fy0 + fx1*fy0 + fx1*fy0 + fx1*fy1)
                    + w101 * (fx0*fz0 + fx1*fz0 + fx1*fz0 + fx1*fz1)
                    + w011 * (fy0*fz0 + fy1*fz0 + fy1*fz0 + fy1*fz1);
            op(*out, w000);
        };

        setdiag(out,         kernel);
        setdiag(out + osc,   kernel + 10);
        setdiag(out + osc*2, kernel + 20);
    }

    //------------------------------------------------------------------
    //                          LAME
    //------------------------------------------------------------------

    static const int kernelsize_lame = 13;

    /// kernel <- [
    ///      absx, wx100, wx010, wx001,
    ///      absy, wy100, wy010, wy001,
    ///      absz, wz100, wz010, wz001,
    ///      ww]
    __device__ static inline  void
    make_kernel_lame(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);

        reduce_t w100 = - membrane * vx;
        reduce_t w010 = - membrane * vy;
        reduce_t w001 = - membrane * vz;
        reduce_t w000 = absolute;

        kernel[0]  = w000 / vx;
        kernel[1]  = w100 / vx - 2*shears - div;
        kernel[2]  = w010 / vx - shears*(vy/vx);
        kernel[3]  = w001 / vx - shears*(vz/vx);

        kernel[4]  = w000 / vy;
        kernel[5]  = w100 / vy - shears*(vx/vy);
        kernel[6]  = w010 / vy - 2*shears - div;
        kernel[7]  = w001 / vy - shears*(vz/vy);

        kernel[8]  = w000 / vz;
        kernel[9]  = w100 / vz - shears*(vx/vz);
        kernel[10] = w010 / vz - shears*(vy/vz);
        kernel[11] = w001 / vz - 2*shears - div;

        kernel[12] = 0.25 * (shears + div);
    }

    /// kernel <- [
    ///      wx000, wx100, wx010, wx001,
    ///      wy000, wy100, wy010, wy001,
    ///      wz000, wz100, wz010, wz001,
    ///      ww]
    __device__ static inline  void
    make_fullkernel_lame(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);

        reduce_t w100 = - membrane * vx;
        reduce_t w010 = - membrane * vy;
        reduce_t w001 = - membrane * vz;
        reduce_t w000 = absolute - 2 * (w100 + w010 + w001);

        kernel[0]  = w000 / vx + 2*shears*(2*vx+vy+vz)/vx + 2*div;
        kernel[1]  = w100 / vx - 2*shears - div;
        kernel[2]  = w010 / vx - shears*(vy/vx);
        kernel[3]  = w001 / vx - shears*(vz/vx);

        kernel[4]  = w000 / vy + 2*shears*(vx+2*vy+vz)/vy + 2*div;
        kernel[5]  = w100 / vy - shears*(vx/vy);
        kernel[6]  = w010 / vy - 2*shears - div;
        kernel[7]  = w001 / vy - shears*(vz/vy);

        kernel[8]  = w000 / vz + 2*shears*(vx+vy+2*vz)/vz + 2*div;
        kernel[9]  = w100 / vz - shears*(vx/vz);
        kernel[10] = w010 / vz - shears*(vy/vz);
        kernel[11] = w001 / vz - 2*shears - div;

        kernel[12] = 0.25 * (shears + div);
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_lame(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[3], const offset_t size[3],
        const offset_t stride[3], offset_t osc, offset_t isc,
        const reduce_t kernel[13])
    {
        offset_t  x = loc[0],     y = loc[1],     z = loc[2];
        offset_t nx = size[0],   ny = size[1],   nz = size[2];
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];

        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        signed char fy0  = bound_utils_y::sign(y-1, ny);
        signed char fy1  = bound_utils_y::sign(y+1, ny);
        signed char fz0  = bound_utils_z::sign(z-1, nz);
        signed char fz1  = bound_utils_z::sign(z+1, nz);
        offset_t    x0  = (bound_utils_x::index(x-1, nx) - x) * sx;
        offset_t    x1  = (bound_utils_x::index(x+1, nx) - x) * sx;
        offset_t    y0  = (bound_utils_y::index(y-1, ny) - y) * sy;
        offset_t    y1  = (bound_utils_y::index(y+1, ny) - y) * sy;
        offset_t    z0  = (bound_utils_z::index(z-1, nz) - z) * sz;
        offset_t    z1  = (bound_utils_z::index(z+1, nz) - z) * sz;

        reduce_t wx000 = kernel[0], wx100 = kernel[1], wx010 = kernel[2],  wx001 = kernel[3],
                 wy000 = kernel[4], wy100 = kernel[5], wy010 = kernel[6],  wy001 = kernel[7],
                 wz000 = kernel[8], wz100 = kernel[9], wz010 = kernel[10], wz001 = kernel[11],
                 w2    = kernel[12];

        reduce_t center0 = static_cast<reduce_t>(inp[0]),
                 center1 = static_cast<reduce_t>(inp[isc]),
                 center2 = static_cast<reduce_t>(inp[isc*2]);

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
        auto cget2 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc*2, o, f) - center2;
        };
        auto get2 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc*2, o, f);
        };

        op(out[0],
              wx000 * center0
            + wx100 * (cget0(x0, fx0) + cget0(x1, fx1))
            + wx010 * (cget0(y0, fy0) + cget0(y1, fy1))
            + wx001 * (cget0(z0, fz0) + cget0(z1, fz1))
            + w2 * (
                  get1(x1+y0, fx1*fy0) + get1(x0+y1, fx0*fy1)
                - get1(x0+y0, fx0*fy0) - get1(x1+y1, fx1*fy1)
                + get2(x1+z0, fx1*fz0) + get2(x0+z1, fx0*fz1)
                - get2(x0+z0, fx0*fz0) - get2(x1+z1, fx1*fz1)
            )
        );

        op(out[osc],
              wy000 * center1
            + wy100 * (cget1(x0, fx0) + cget1(x1, fx1))
            + wy010 * (cget1(y0, fy0) + cget1(y1, fy1))
            + wy001 * (cget1(z0, fz0) + cget1(z1, fz1))
            + w2 * (
                  get0(x1+y0, fx1*fy0) + get0(x0+y1, fx0*fy1)
                - get0(x0+y0, fx0*fy0) - get0(x1+y1, fx1*fy1)
                + get2(y1+z0, fy1*fz0) + get2(y0+z1, fy0*fz1)
                - get2(y0+z0, fy0*fz0) - get2(y1+z1, fy1*fz1)
            )
        );

        op(out[osc*2],
              wz000 * center2
            + wz100 * (cget2(x0, fx0) + cget2(x1, fx1))
            + wz010 * (cget2(y0, fy0) + cget2(y1, fy1))
            + wz001 * (cget2(z0, fz0) + cget2(z1, fz1))
            + w2 * (
                  get0(x1+z0, fx1*fz0) + get0(x0+z1, fx0*fz1)
                - get0(x0+z0, fx0*fz0) - get0(x1+z1, fx1*fz1)
                + get1(y1+z0, fy1*fz0) + get1(y0+z1, fy0*fz1)
                - get1(y0+z0, fy0*fz0) - get1(y1+z1, fy1*fz1)
            )
        );
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
     kernel_lame(
        scalar_t * out, const offset_t sc[2], const offset_t stride[3],
        const reduce_t kernel[13])
    {
        offset_t sc0 = sc[0], sc1 = sc[1];
        const offset_t sx = stride[0], sy = stride[1], sz = stride[2];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0], w100 = kernel[1],
                     w010 = kernel[2], w001 = kernel[3];
            op(out[0],      w000);
            op(out[-sx],    w100);
            op(out[+sx],    w100);
            op(out[-sy],    w010);
            op(out[+sy],    w010);
            op(out[-sz],    w001);
            op(out[+sz],    w001);
        };

        auto xxout = out, yyout = out + sc0 + sc1, zzout = out + (sc0 + sc1)*2;
        setkernel(xxout, kernel);
        setkernel(yyout, kernel + 4);
        setkernel(zzout, kernel + 8);

        auto w2 = kernel[12];
        auto xyout = out + sc0,         yxout = out + sc1,
             xzout = out + sc0*2,       zxout = out + sc1*2,
             yzout = out + sc0*2 + sc1, zyout = out + sc0 + sc1*2;
        op(xyout[+sx+sy], -w2); op(xyout[-sx-sy], -w2); op(yxout[+sx+sy], -w2); op(yxout[-sx-sy], -w2);
        op(xyout[-sx+sy], +w2); op(xyout[+sx-sy], +w2); op(yxout[-sx+sy], +w2); op(yxout[+sx-sy], +w2);
        op(xzout[+sx+sz], -w2); op(xzout[-sx-sz], -w2); op(zxout[+sx+sz], -w2); op(zxout[-sx-sz], -w2);
        op(xzout[-sx+sz], +w2); op(xzout[+sx-sz], +w2); op(zxout[-sx+sz], +w2); op(zxout[+sx-sz], +w2);
        op(yzout[+sy+sz], -w2); op(yzout[-sy-sz], -w2); op(zyout[+sy+sz], -w2); op(zyout[-sy-sz], -w2);
        op(yzout[-sy+sz], +w2); op(yzout[+sy-sz], +w2); op(zyout[-sy+sz], +w2); op(zyout[+sy-sz], +w2);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline  void
    diag_lame(
        scalar_t * out, offset_t osc,
        const offset_t loc[3], const offset_t size[3],
        const reduce_t kernel[13])
    {
        offset_t  x = loc[0],     y = loc[1],     z = loc[2];
        offset_t nx = size[0],   ny = size[1],   nz = size[2];

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        signed char fy0 = bound_utils_y::sign(y-1, ny);
        signed char fy1 = bound_utils_y::sign(y+1, ny);
        signed char fz0 = bound_utils_z::sign(z-1, nz);
        signed char fz1 = bound_utils_z::sign(z+1, nz);

        reduce_t w2 = kernel[12];

        auto setdiag = [&](scalar_t & out, const reduce_t * kernel) {
            reduce_t w000 = kernel[0], w100 = kernel[1],
                     w010 = kernel[2], w001 = kernel[3];
            w000 -= w100 * (fx0+fx1) + w010 * (fy0+fy1) + w001 * (fz0+fz1);
            op(out, w000);
        };

        setdiag(out[0],     kernel);
        setdiag(out[osc],   kernel + 4);
        setdiag(out[osc*2], kernel + 8);
    }

    //------------------------------------------------------------------
    //                         ABSOLUTE JRLS
    //------------------------------------------------------------------

    // --- vel2mom ---

    template <OpType op = set>
    static inline __device__
    void vel2mom_absolute_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t osc, offset_t isc, const reduce_t kernel[3])
    {
        reduce_t w = static_cast<reduce_t>(*wgt);
        op(out[0],     kernel[0] * w * static_cast<reduce_t>(inp[0]));
        op(out[osc],   kernel[1] * w * static_cast<reduce_t>(inp[isc]));
        op(out[osc*2], kernel[2] * w * static_cast<reduce_t>(inp[isc*2]));
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_absolute_jrls(
        scalar_t * out, const scalar_t * wgt,
        offset_t osc, const reduce_t kernel[3])
    {
        reduce_t w = static_cast<reduce_t>(*wgt);
        op(out[0],     kernel[0] * w);
        op(out[osc],   kernel[1] * w);
        op(out[osc*2], kernel[2] * w);
    }

    //------------------------------------------------------------------
    //                         MEMBRANE JRLS
    //------------------------------------------------------------------

    static const int kernelsize_membrane_jrls = kernelsize_membrane;

    __device__ static inline void
    make_kernel_membrane_jrls(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[3])
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
        const offset_t loc[3], const offset_t size[3],
        const offset_t istride[3], const offset_t wstride[3],
        offset_t osc, offset_t isc, const reduce_t kernel[12])
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t isx = istride[0], isy = istride[1], isz = istride[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

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
        reduce_t w110 = wget(wz0);
        reduce_t w112 = wget(wz1);

        // --- convolution ---

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t m000 = kernel[0], m100 = kernel[1],
                     m010 = kernel[2], m001 = kernel[3];

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
               (m000*w111*2)*center
               + (m100*w011)*get(ix0, fx0) + (m100*w211)*get(ix1, fx1)
               + (m010*w101)*get(iy0, fy0) + (m010*w121)*get(iy1, fy1)
               + (m001*w110)*get(iz0, fz0) + (m001*w112)*get(iz1, fz1)
            );
        };

        conv(out,         inp,         kernel);
        conv(out + osc,   inp + isc,   kernel + 4);
        conv(out + osc*2, inp + isc*2, kernel + 8);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_membrane_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[3], const offset_t size[3],
        const offset_t wstride[3], offset_t osc, const reduce_t kernel[12])
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

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
        reduce_t w110 = wget(iz0) * fz0;
        reduce_t w112 = wget(iz1) * fz1;

        // --- convolution ---

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t m000 = kernel[0], m100 = kernel[1],
                     m010 = kernel[2], m001 = kernel[3];
            op(*out,
               m000*w111*2
               - m100*(w011 + w211)
               - m010*(w101 + w121)
               - m001*(w110 + w112)
            );
        };

        conv(out,         kernel);
        conv(out + osc,   kernel + 4);
        conv(out + osc*2, kernel + 8);
    }

    //------------------------------------------------------------------
    //                         BENDING JRLS
    //------------------------------------------------------------------

    static const int kernelsize_bending_jrls = 13;

    /* kernel = [
     *      lx, ly, lz,
     *      k000, k100, k010, k001, k200, k020, k002, k110, k101, k011]
     *
     *      k000 = absolute
     *      k100 = -4 * lx * (lx + ly + lz) * bending - lx * membrane
     *      k010 = -4 * ly * (lx + ly + lz) * bending - ly * membrane
     *      k001 = -4 * lz * (lx + ly + lz) * bending - lz * membrane
     *      k200 = lx * lx * bending
     *      k020 = ly * ly * bending
     *      k002 = lz * lz * bending
     *      k110 = 2 * lx * ly * bending
     *      k101 = 2 * lx * lz * bending
     *      k011 = 2 * ly * lz * bending
     *
     * where lx = 1/(vx[0]*vx[0])
     *       ly = 1/(vx[1]*vx[1])
     *       lz = 1/(vx[2]*vx[2])
     */
    static inline __device__ void
    make_kernel_bending_jrls(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);
        reduce_t vxyz = vx + vy + vz;

        kernel[0] = vx;
        kernel[1] = vy;
        kernel[2] = vz;

        kernel[3]  = absolute;
        kernel[4]  = (-4 * vxyz * bending - membrane) * vx;
        kernel[5]  = (-4 * vxyz * bending - membrane) * vy;
        kernel[6]  = (-4 * vxyz * bending - membrane) * vz;
        kernel[7]  = vx * vx * bending;
        kernel[8]  = vy * vy * bending;
        kernel[9]  = vz * vz * bending;
        kernel[10] = 2 * vx * vy * bending;
        kernel[11] = 2 * vx * vz * bending;
        kernel[12] = 2 * vy * vz * bending;

        for (int k=4; k<kernelsize_bending_jrls; ++k)
            kernel[k] *= 0.25;
    }

    // --- vel2mom ---

    template <OpType op = set>
    __device__ static inline void
    vel2mom_bending_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[3], const offset_t size[3],
        const offset_t istride[3], const offset_t wstride[3],
        offset_t osc, offset_t isc, const reduce_t kernel[13])
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t isx = istride[0], isy = istride[1], isz = istride[2];
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
        ix0 *= isx;
        ix1 *= isx;
        ix3 *= isx;
        ix4 *= isx;
        iy0 *= isy;
        iy1 *= isy;
        iy3 *= isy;
        iy4 *= isy;
        iz0 *= isz;
        iz1 *= isz;
        iz3 *= isz;
        iz4 *= isz;

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

        reduce_t w122 = wget(wx1);
        reduce_t w322 = wget(wx3);
        reduce_t w212 = wget(wy1);
        reduce_t w232 = wget(wy3);
        reduce_t w221 = wget(wz1);
        reduce_t w223 = wget(wz3);

        reduce_t w022 = wget(wx0);
        reduce_t w422 = wget(wx4);
        reduce_t w202 = wget(wy0);
        reduce_t w242 = wget(wy4);
        reduce_t w220 = wget(wz0);
        reduce_t w224 = wget(wz4);

        reduce_t w112 = wget(wx1+wy1);
        reduce_t w132 = wget(wx1+wy3);
        reduce_t w312 = wget(wx3+wy1);
        reduce_t w332 = wget(wx3+wy3);
        reduce_t w121 = wget(wx1+wz1);
        reduce_t w123 = wget(wx1+wz3);
        reduce_t w321 = wget(wx3+wz1);
        reduce_t w323 = wget(wx3+wz3);
        reduce_t w211 = wget(wy1+wz1);
        reduce_t w213 = wget(wy1+wz3);
        reduce_t w231 = wget(wy3+wz1);
        reduce_t w233 = wget(wy3+wz3);

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

    //------------------------------------------------------------------
    //                           LAME JRLS
    //------------------------------------------------------------------

    static const int kernelsize_lame_jrls = 14;

    /* kernel = [wx000, wx100, wx010, wx001,
     *           wy000, wy100, wy010, wy001,
     *           wz000, wz100, wz010, wz001,
     *           d2, s2]
     *
     * wx100 = -(0.5*div + shears)
     * wx010 = -0.5*shears*(vy/vx)
     * wx001 = -0.5*shears*(vz/vx)
     * wy100 = -0.5*shears*(vx/vy)
     * wy010 = -(0.5*div + shears)
     * wy001 = -0.5*shears*(vz/vy)
     * wz100 = -0.5*shears*(vx/vz)
     * wz010 = -0.5*shears*(vy/vz)
     * wz001 = -(0.5*div + shears)
     * d2    = 0.25*div
     * s2    = 0.25*shears
     */
    __device__ static inline void
    make_kernel_lame_jrls(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[3])
    {
        make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

        for (int k=0; k < 12; ++k)
            kernel[k] *= 0.5;

        kernel[12] = 0.25 * shears;
        kernel[13] = 0.25 * div;
    }

    // --- vel2mom ---

    template <OpType op = set>
    static inline __device__
    void vel2mom_lame_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[3], const offset_t size[3],
        const offset_t istride[3], const offset_t wstride[3],
        offset_t osc, offset_t isc, const reduce_t kernel[14])
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t isx = istride[0], isy = istride[1], isz = istride[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

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
        reduce_t w110 = wget(wz0);
        reduce_t w112 = wget(wz1);

        // --- weight map kernel

        reduce_t wx000 = kernel[0],  wx100 = kernel[1],  wx010 = kernel[2],  wx001 = kernel[3],
                 wy000 = kernel[4],  wy100 = kernel[5],  wy010 = kernel[6],  wy001 = kernel[7],
                 wz000 = kernel[8],  wz100 = kernel[9],  wz010 = kernel[10], wz001 = kernel[11],
                 d2    = kernel[12], s2    = kernel[13];

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

        reduce_t center2 = static_cast<reduce_t>(inp[isc*2]);
        auto cget2 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc*2, o, f) - center2;
        };
        auto get2 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp + isc*2, o, f);
        };

        op(out[0],
           (wx000*w111*2)*center0
           + (wx100*(w011+w111))*cget0(ix0, fx0) + (wx100*(w211+w111))*cget0(ix1, fx1)
           + (wx010*(w101+w111))*cget0(iy0, fy0) + (wx010*(w121+w111))*cget0(iy1, fy1)
           + (wx001*(w110+w111))*cget0(iz0, fz0) + (wx001*(w112+w111))*cget0(iz1, fz1)
           - (d2*w011 + s2*w101)*get1(ix0+iy0, fx0*fy0)
           - (d2*w211 + s2*w121)*get1(ix1+iy1, fx1*fy1)
           + (d2*w011 + s2*w121)*get1(ix0+iy1, fx0*fy1)
           + (d2*w211 + s2*w101)*get1(ix1+iy0, fx1*fy0)
           - (d2*w011 + s2*w110)*get2(ix0+iz0, fx0*fz0)
           - (d2*w211 + s2*w112)*get2(ix1+iz1, fx1*fz1)
           + (d2*w011 + s2*w112)*get2(ix0+iz1, fx0*fz1)
           + (d2*w211 + s2*w110)*get2(ix1+iz0, fx1*fz0)
        );

        op(out[osc],
           (wy000*w111*2)*center1
           + (wy100*(w011+w111))*cget1(ix0, fx0) + (wy100*(w211+w111))*cget1(ix1, fx1)
           + (wy010*(w101+w111))*cget1(iy0, fy0) + (wy010*(w121+w111))*cget1(iy1, fy1)
           + (wy001*(w110+w111))*cget1(iz0, fz0) + (wy001*(w112+w111))*cget1(iz1, fz1)
           - (d2*w101 + s2*w011)*get0(iy0+ix0, fy0*fx0)
           - (d2*w121 + s2*w211)*get0(iy1+ix1, fy1*fx1)
           + (d2*w101 + s2*w211)*get0(iy0+ix1, fy0*fx1)
           + (d2*w121 + s2*w011)*get0(iy1+ix0, fy1*fx0)
           - (d2*w101 + s2*w110)*get2(iy0+iz0, fy0*fz0)
           - (d2*w121 + s2*w112)*get2(iy1+iz1, fy1*fz1)
           + (d2*w101 + s2*w112)*get2(iy0+iz1, fy0*fz1)
           + (d2*w121 + s2*w110)*get2(iy1+iz0, fy1*fz0)
        );

        op(out[osc*2],
           (wz000*w111*2)*center2
           + (wz100*(w011+w111))*cget2(ix0, fx0) + (wz100*(w211+w111))*cget2(ix1, fx1)
           + (wz010*(w101+w111))*cget2(iy0, fy0) + (wz010*(w121+w111))*cget2(iy1, fy1)
           + (wz001*(w110+w111))*cget2(iz0, fz0) + (wz001*(w112+w111))*cget2(iz1, fz1)
           - (d2*w110 + s2*w011)*get0(iz0+ix0, fz0*fx0)
           - (d2*w112 + s2*w211)*get0(iz1+ix1, fz1*fx1)
           + (d2*w110 + s2*w211)*get0(iz0+ix1, fz0*fx1)
           + (d2*w112 + s2*w011)*get0(iz1+ix0, fz1*fx0)
           - (d2*w110 + s2*w101)*get1(iz0+iy0, fz0*fy0)
           - (d2*w112 + s2*w121)*get1(iz1+iy1, fz1*fy1)
           + (d2*w110 + s2*w121)*get1(iz0+iy1, fz0*fy1)
           + (d2*w112 + s2*w101)*get1(iz1+iy0, fz1*fy0)
        );
    }

    // --- diagonal ---

    template <OpType op = set>
    static inline __device__
    void diag_lame_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[3], const offset_t size[3],
        const offset_t wstride[3], offset_t osc, const reduce_t kernel[14])
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

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

        // --- load weight map ---

        reduce_t w0 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o) + w0;
        };

        reduce_t wx = wget(ix0) * fx0 + wget(ix1) * fx1;
        reduce_t wy = wget(iy0) * fy0 + wget(iy1) * fy1;
        reduce_t wz = wget(iz0) * fz0 + wget(iz1) * fz1;

        // --- compute convolution ---

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t m000 = kernel[0], m100 = kernel[1],
                     m010 = kernel[2], m001 = kernel[3];
            op(*out, m000*w0*2 - m100*wx - m010*wy - m001*wz);
        };

        conv(out,         kernel);
        conv(out + osc,   kernel + 4);
        conv(out + osc*2, kernel + 8);
    }

};

} // namespace reg_grid
} // namespace jf

#endif // JF_REGULARISERS_GRID_3D
