#ifndef JF_REGULARISERS_FIELD_2D
#define JF_REGULARISERS_FIELD_2D
#include "../../cuda_switch.h"
#include "../../bounds.h"
#include "../../utils.h"
#include "utils.h"

namespace jf {
namespace reg_field {

//----------------------------------------------------------------------
//          low-level kernels for anything regularization
//----------------------------------------------------------------------

template <int C,
          typename scalar_t, typename reduce_t, typename offset_t,
          bound::type BX, bound::type BY>
struct RegField<C, two, scalar_t, reduce_t, offset_t, BX, BY> {
    static const int D = two;
    using bound_utils_x = bound::utils<BX>;
    using bound_utils_y = bound::utils<BY>;
    typedef scalar_t & (*OpType)(scalar_t &, const reduce_t &);

    //------------------------------------------------------------------
    //                            ABSOLUTE
    //------------------------------------------------------------------

    static const int kernelsize_absolute = C;

    /// kernel <- [abs, ...]
    __device__ static inline void
    make_kernel_absolute(reduce_t kernel[C], const reduce_t absolute[C])
    {
#       pragma unroll
        for (int c = 0; c < C; ++c)
            kernel[c] = absolute[c];
    }

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_absolute(
        scalar_t * out, const scalar_t * inp,
        offset_t osc, offset_t isc, const reduce_t kernel[C])
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
            op(out[osc*c], kernel[c] * inp[isc*c]);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline  void
    kernel_absolute(scalar_t * out, offset_t osc, const reduce_t kernel[C])
    {
#       pragma unroll
        for (offset_t c = 0; c < C; ++c)
            op(out[osc*c], kernel[c]);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline  void
    diag_absolute(scalar_t * out, offset_t osc, const reduce_t kernel[C])
    {
        return kernel_absolute(out, osc, kernel);
    }

    //------------------------------------------------------------------
    //                            MEMBRANE
    //------------------------------------------------------------------

    static const int kernelsize_membrane = (D+1)*C;

    /// kernel <- [abs, w10, w01, ...]
    __device__ static inline void
    make_kernel_membrane(
        reduce_t * kernel,
        const reduce_t absolute[C],
        const reduce_t membrane[C],
        const reduce_t voxel_size[D])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);
        for (int c = 0; c < C; ++c, kernel+=(D+1))
        {
            reduce_t m = membrane[c];
            kernel[0] = absolute[c];
            kernel[1] = -m * vx;
            kernel[2] = -m * vy;
        }
    }

    /// kernel <- [w00, w10, w01, ...]
    __device__ static inline void
    make_fullkernel_membrane(
        reduce_t * kernel,
        const reduce_t absolute[C],
        const reduce_t membrane[C],
        const reduce_t voxel_size[D])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);
        for (int c = 0; c < C; ++c, kernel+=(D+1))
        {
            reduce_t m = membrane[c];
            kernel[0] = absolute[c] + 2 * m * (vx + vy);
            kernel[1] = -m * vx;
            kernel[2] = -m * vy;
        }
    }

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_membrane(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[D], const offset_t size[D], const offset_t stride[D],
        offset_t osc, offset_t isc, const reduce_t kernel[(D+1)*C])
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

        for (offset_t c=0; c<C; ++c)
            conv(out + osc*c, inp + isc*c, kernel + (D+1)*c);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
    kernel_membrane(
        scalar_t * out, offset_t sc, const offset_t stride[3],
        const reduce_t kernel[(D+1)*C])
    {
        offset_t sx = stride[0], sy = stride[1];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w00 = kernel[0], w10 = kernel[1], w01 = kernel[2];
            op(out[0],   w00);
            op(out[-sx], w10);
            op(out[+sx], w10);
            op(out[-sy], w01);
            op(out[+sy], w01);
        };

        for (offset_t c=0; c<C; ++c)
            setkernel(out + sc*c, kernel + (D+1)*c);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_membrane(
        scalar_t * out, offset_t osc,
         const offset_t loc[D], const offset_t size[D],
         const reduce_t kernel[(D+1)*C])
    {
        offset_t  x = loc[0],   y = loc[1];
        offset_t nx = size[0], ny = size[1];

        signed char fx = bound_utils_x::sign(x-1, nx)
                       + bound_utils_x::sign(x+1, nx);
        signed char fy = bound_utils_y::sign(y-1, ny)
                       + bound_utils_y::sign(y+1, ny);

        for (offset_t c=0; c<C; ++c, kernel += (D+1))
            op(out[osc*c], kernel[0] - kernel[1]*fx - kernel[2]*fy);
    }

    //------------------------------------------------------------------
    //                            BENDING
    //------------------------------------------------------------------

    static const int kernelsize_bending = 6*C;

    /// kernel <- [abs, w10, w01, w20, w02, w11, ...]
    __device__ static inline void
    make_kernel_bending(
        reduce_t * kernel,
        const reduce_t absolute[C],
        const reduce_t membrane[C],
        const reduce_t bending[C],
        const reduce_t voxel_size[D])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);
        for (int c=0; c<C; ++c, kernel+=6)
        {
            reduce_t m = membrane[c], b = bending[c];
            kernel[0] = absolute[c];
            kernel[1] = -4 * b * vx * (vx + vy) - m * vx;
            kernel[2] = -4 * b * vy * (vx + vy) - m * vy;
            kernel[3] = b * vx * vx;
            kernel[4] = b * vy * vy;
            kernel[5] = 2 * b * vx * vy;
        }
    }

    /// kernel <- [w00, w10, w01, w20, w02, w11, ...]
    __device__ static inline void
    make_fullkernel_bending(
        reduce_t * kernel,
        const reduce_t absolute[C],
        const reduce_t membrane[C],
        const reduce_t bending[C],
        const reduce_t voxel_size[D])
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1];
        vx = 1./(vx*vx); vy = 1./(vy*vy);
        for (int c=0; c<C; ++c, kernel+=6)
        {
            reduce_t m = membrane[c], b = bending[c];
            kernel[1] = -4 * b * vx * (vx + vy) - m * vx;
            kernel[2] = -4 * b * vy * (vx + vy) - m * vy;
            kernel[3] = b * vx * vx;
            kernel[4] = b * vy * vy;
            kernel[5] = 2 * b * vx * vy;
            kernel[0] = absolute[c]
                      - 2 * (kernel[1] + kernel[2] + kernel[3] + kernel[4])
                      - 4 * kernel[5];
        }
    }

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_bending(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[D], const offset_t size[D],
        const offset_t stride[D], offset_t osc, offset_t isc,
        const reduce_t kernel[6*C])
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
            reduce_t w00 = kernel[0],
                     w10 = kernel[1], w01 = kernel[2],
                     w20 = kernel[3], w02 = kernel[4],
                     w11 = kernel[5];

            reduce_t center = static_cast<reduce_t>(inp[0]);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
                  w00 * center
                + w10 * (get(x0, fx0) + get(x1, fx1))
                + w01 * (get(y0, fy0) + get(y1, fy1))
                + w20 * (get(x00, fx00) + get(x11, fx11))
                + w02 * (get(y00, fy00) + get(y11, fy11))
                + w11 * (get(x0+y0, fx0*fy0) + get(x1+y0, fx1*fy0) +
                         get(x0+y1, fx0*fy1) + get(x1+y1, fx1*fy1))
            );
        };

        for (offset_t c=0; c<C; ++c)
            conv(out + osc*c, inp + isc*c, kernel + 6*c);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
     kernel_bending(
        scalar_t * out, offset_t sc, const offset_t stride[D],
        const reduce_t kernel[6*C])
    {
        offset_t sx = stride[0], sy = stride[1];

        auto setkernel = [&](scalar_t * o, const reduce_t * ker)
        {
            reduce_t w00 = ker[0],
                     w10 = ker[1], w01 = ker[2],
                     w20 = ker[3], w02 = ker[4],
                     w11 = ker[5];
            op(o[0],      w00);
            op(o[-sx],    w10);
            op(o[+sx],    w10);
            op(o[-sy],    w01);
            op(o[+sy],    w01);
            op(o[-sx*2],  w20);
            op(o[+sx*2],  w20);
            op(o[-sy*2],  w02);
            op(o[+sy*2],  w02);
            op(o[-sx-sy], w11);
            op(o[-sx+sy], w11);
            op(o[+sx-sy], w11);
            op(o[+sx+sy], w11);
        };

        for (offset_t c=0; c<C; ++c)
            setkernel(out + sc*c, kernel + 6*c);
    }

    // --- diagonal ---

    template <OpType op = set>
    static inline __device__ void
    diag_bending(
        scalar_t * out, offset_t osc,
        const offset_t loc[D], const offset_t size[D],
        const reduce_t kernel[6*C])
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

        auto setdiag = [&](scalar_t & out, const reduce_t * kernel)
        {
            reduce_t w00 = kernel[0],
                     w10 = kernel[1], w01 = kernel[2],
                     w20 = kernel[3], w02 = kernel[4],
                     w11 = kernel[5];
            w00 -=   w10 * (fx0 + fx1)   + w01 * (fy0 + fy1)
                   + w20 * (fx00 + fx11) + w02 * (fy00 + fy11)
                   + w11 * (fx0*fy0 + fx1*fy0 + fx1*fy0 + fx1*fy1);
            op(out, w00);
        };

        for (offset_t c=0; c<C; ++c)
            setdiag(out[osc*c], kernel + 6*c);
    }

    //------------------------------------------------------------------
    //                         ABSOLUTE RLS
    //------------------------------------------------------------------

    // --- matvec ---

    template <OpType op = set>
    static inline __device__
    void matvec_absolute_rls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t osc, offset_t isc, offset_t wsc, const reduce_t kernel[C])
    {
        for (offset_t c=0; c<C; ++c)
            op(out[osc*c], kernel[c] *
                           static_cast<reduce_t>(wgt[wsc*c]) *
                           static_cast<reduce_t>(inp[isc*c]));
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_absolute_rls(
        scalar_t * out, const scalar_t * wgt,
        offset_t osc, offset_t wsc, const reduce_t kernel[C])
    {
        for (offset_t c=0; c<C; ++c)
            op(out[osc*c], kernel[c] * static_cast<reduce_t>(wgt[wsc*c]));
    }

    //------------------------------------------------------------------
    //                         ABSOLUTE JRLS
    //------------------------------------------------------------------

    // --- matvec ---

    template <OpType op = set>
    static inline __device__
    void matvec_absolute_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t osc, offset_t isc, const reduce_t kernel[C])
    {
        reduce_t w = static_cast<reduce_t>(*wgt);
        for (offset_t c=0; c<C; ++c)
            op(out[osc*c], kernel[c] * w * static_cast<reduce_t>(inp[isc*c]));
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_absolute_jrls(
        scalar_t * out, const scalar_t * wgt,
        offset_t osc, const reduce_t kernel[3])
    {
        reduce_t w = static_cast<reduce_t>(*wgt);
        for (offset_t c=0; c<C; ++c)
            op(out[osc*c], kernel[c] * w);
    }

    //------------------------------------------------------------------
    //                         MEMBRANE RLS
    //------------------------------------------------------------------

    static const int kernelsize_membrane_rls = kernelsize_membrane;

    __device__ static inline void
    make_kernel_membrane_rls(
        reduce_t * kernel,
        const reduce_t absolute[C],
        const reduce_t membrane[C],
        const reduce_t voxel_size[D])
    {
        make_kernel_membrane(kernel, absolute, membrane, voxel_size);
        for (int k=0; k<kernelsize_membrane_rls; ++k)
            kernel[k] *= 0.5;
    }

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_membrane_rls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t istride[D], const offset_t wstride[D],
        offset_t osc, offset_t isc, offset_t wsc, const reduce_t kernel[(D+1)*C])
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

        for (offset_t c=0; c<C; ++c)
        {
            // --- load weight map ---
            reduce_t w11 = static_cast<reduce_t>(wgt[wsc*c]);
            auto wget = [&](offset_t o)
            {
                return bound::cget<reduce_t>(wgt + wsc*c, o) + w11;
            };
            reduce_t w01 = wget(wx0);
            reduce_t w21 = wget(wx1);
            reduce_t w10 = wget(wy0);
            reduce_t w12 = wget(wy1);

            // --- convolution ---

            auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
            {
                reduce_t m00 = kernel[0], m10 = kernel[1], m01 = kernel[2];

                reduce_t center = static_cast<reduce_t>(*inp);
                auto get = [&](offset_t o, signed char f)
                {
                    return bound::cget<reduce_t>(inp, o, f) - center;
                };

                op(*out,
                   (m00*w11*2)*center
                   + (m10*w01)*get(ix0, fx0) + (m10*w21)*get(ix1, fx1)
                   + (m01*w10)*get(iy0, fy0) + (m01*w12)*get(iy1, fy1)
                );
            };

            conv(out + osc*c, inp + isc*c, kernel + (D+1)*c);
        }
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_membrane_rls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t wstride[D], offset_t osc, offset_t wsc,
        const reduce_t kernel[(D+1)*C])
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

        for (offset_t c=0; c<C; ++c)
        {
            // --- load weight map ---
            reduce_t w11 = static_cast<reduce_t>(wgt[wsc*c]);
            auto wget = [&](offset_t o)
            {
                return bound::cget<reduce_t>(wgt + wsc*c, o) + w11;
            };
            reduce_t w01 = wget(ix0) * fx0;
            reduce_t w21 = wget(ix1) * fx1;
            reduce_t w10 = wget(iy0) * fy0;
            reduce_t w12 = wget(iy1) * fy1;

            // --- convolution ---

            auto conv = [&](scalar_t * out, const reduce_t * kernel)
            {
                reduce_t m00 = kernel[0], m10 = kernel[1], m01 = kernel[2];
                op(*out, m00*w11*2 - m10*(w01 + w21) - m01*(w10 + w12));
            };

            conv(out + osc*c, kernel + (D+1)*c);
        }
    }

    //------------------------------------------------------------------
    //                         MEMBRANE JRLS
    //------------------------------------------------------------------

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_membrane_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t istride[D], const offset_t wstride[D],
        offset_t osc, offset_t isc, const reduce_t kernel[(D+1)*C])
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
        reduce_t w11 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o) + w11;
        };
        reduce_t w01 = wget(wx0);
        reduce_t w21 = wget(wx1);
        reduce_t w10 = wget(wy0);
        reduce_t w12 = wget(wy1);

        // --- convolution ---

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t m00 = kernel[0], m10 = kernel[1], m01 = kernel[2];

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
               (m00*w11*2)*center
               + (m10*w01)*get(ix0, fx0) + (m10*w21)*get(ix1, fx1)
               + (m01*w10)*get(iy0, fy0) + (m01*w12)*get(iy1, fy1)
            );
        };

        for (offset_t c=0; c<C; ++c)
            conv(out + osc*c, inp + isc*c, kernel + (D+1)*c);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_membrane_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t wstride[D], offset_t osc,
        const reduce_t kernel[(D+1)*C])
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
        reduce_t w11 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o) + w11;
        };
        reduce_t w01 = wget(ix0) * fx0;
        reduce_t w21 = wget(ix1) * fx1;
        reduce_t w10 = wget(iy0) * fy0;
        reduce_t w12 = wget(iy1) * fy1;

        // --- convolution ---

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t m00 = kernel[0], m10 = kernel[1],  m01 = kernel[2];
            op(*out, m00*w11*2 - m10*(w01 + w21) - m01*(w10 + w12));
        };

        for (offset_t c=0; c<C; ++c)
            conv(out + osc*c, kernel + (D+1)*c);
    }

    //------------------------------------------------------------------
    //                         BENDING RLS
    //------------------------------------------------------------------

    static const int kernelsize_bending_rls = kernelsize_bending;

    static inline __device__ void
    make_kernel_bending_rls(
        reduce_t * kernel,
        const reduce_t absolute[C],
        const reduce_t membrane[C],
        const reduce_t bending[C],
        const reduce_t voxel_size[D])
    {
        make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);
        for (int k=0; k<kernelsize_bending_rls; ++k)
        {
            if (k % 6 == 0) continue;
            kernel[k] *= 0.25;
        }
    }

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_bending_rls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t istride[D], const offset_t wstride[D],
        offset_t osc, offset_t isc, offset_t wsc, const reduce_t kernel[6*C])
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

        for (offset_t c=0; c<C; ++c, kernel+=6, out+=osc, inp+=isc, wgt+=wsc)
        {
            reduce_t b00 = kernel[0],
                     b10 = kernel[1], b01 = kernel[2],
                     b20 = kernel[3], b02 = kernel[4],
                     b11 = kernel[5];

            reduce_t w22 = static_cast<reduce_t>(*wgt);
            auto wget = [&](offset_t o)
            {
                return bound::cget<reduce_t>(wgt, o);
            };

            // first order neighbours
            reduce_t w12 = wget(wx1);
            reduce_t w32 = wget(wx3);
            reduce_t w21 = wget(wy1);
            reduce_t w23 = wget(wy3);

            // second order neighbours
            reduce_t w02 = wget(wx0);
            reduce_t w42 = wget(wx4);
            reduce_t w20 = wget(wy0);
            reduce_t w24 = wget(wy4);

            // diagonal neighbours
            reduce_t w11 = wget(wx1+wy1);
            reduce_t w13 = wget(wx1+wy3);
            reduce_t w31 = wget(wx3+wy1);
            reduce_t w33 = wget(wx3+wy3);

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            auto sum1 = [&]()
            {
                reduce_t m12 = (b10 - 2*b20) * (w22 + w12)
                               - 2*b20 * (w32 + w02)
                               - b11 * (w21 + w11 + w23 + w13);
                reduce_t m32 = (b10 - 2*b20) * (w22 + w32)
                               - 2*b20 * (w42 + w12)
                               - b11 * (w23 + w33 + w21 + w31);

                reduce_t m21 = (b01 - 2*b02) * (w22 + w21)
                               - 2*b02 * (w23 + w20)
                               - b11 * (w12 + w11 + w32 + w13);
                reduce_t m23 = (b01 - 2*b02) * (w22 + w23)
                               - 2*b02 * (w24 + w21)
                               - b11 * (w32 + w33 + w12 + w31);

                return (m12*get(ix1, fx1) +  m32*get(ix3, fx3) +
                        m21*get(iy1, fy1) +  m23*get(iy3, fy3));
            };

            auto sum2 = [&]()
            {
                reduce_t m02 = b20 * (2 * w12 + w02 + w22);
                reduce_t m42 = b20 * (2 * w32 + w42 + w22);
                reduce_t m20 = b02 * (2 * w21 + w20 + w22);
                reduce_t m24 = b02 * (2 * w23 + w24 + w22);

                return (m02*get(ix0, fx0) +  m42*get(ix4, fx4) +
                        m20*get(iy0, fy0) +  m24*get(iy4, fy4));
            };

            auto sumdiag = [&]()
            {
                reduce_t m11 = b11 * (w22 + w12 + w21 + w11);
                reduce_t m13 = b11 * (w22 + w12 + w23 + w13);
                reduce_t m31 = b11 * (w22 + w32 + w21 + w31);
                reduce_t m33 = b11 * (w22 + w32 + w23 + w33);

                return (m11*get(ix1+iy1, fx1*fy1) +  m13*get(ix1+iy3, fx1*fy3) +
                        m31*get(ix3+iy1, fx3*fy1) +  m33*get(ix3+iy3, fx3*fy3));
            };

            op(*out, b00*center + sum1() + sum2() + sumdiag());
        }
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_bending_rls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t wstride[D], offset_t osc, offset_t wsc,
        const reduce_t kernel[6*C])
    {
        offset_t   x = loc[0],       y = loc[1];
        offset_t  nx = size[0],     ny = size[1];
        offset_t wsx = wstride[0], wsy = wstride[1];

        signed char fx0 = bound_utils_x::sign(x-2, nx);
        signed char fx1 = bound_utils_x::sign(x-1, nx);
        signed char fx3 = bound_utils_x::sign(x+1, nx);
        signed char fx4 = bound_utils_x::sign(x+2, nx);
        signed char fy0 = bound_utils_y::sign(y-2, ny);
        signed char fy1 = bound_utils_y::sign(y-1, ny);
        signed char fy3 = bound_utils_y::sign(y+1, ny);
        signed char fy4 = bound_utils_y::sign(y+2, ny);
        offset_t    ix0 = (bound_utils_x::index(x-2, nx) - x) * wsx;
        offset_t    ix1 = (bound_utils_x::index(x-1, nx) - x) * wsx;
        offset_t    ix3 = (bound_utils_x::index(x+1, nx) - x) * wsx;
        offset_t    ix4 = (bound_utils_x::index(x+2, nx) - x) * wsx;
        offset_t    iy0 = (bound_utils_y::index(y-2, ny) - y) * wsy;
        offset_t    iy1 = (bound_utils_y::index(y-1, ny) - y) * wsy;
        offset_t    iy3 = (bound_utils_y::index(y+1, ny) - y) * wsy;
        offset_t    iy4 = (bound_utils_y::index(y+2, ny) - y) * wsy;

        for (offset_t c=0; c<C; ++c, kernel+=6, out+=osc, wgt+=wsc)
        {

            reduce_t b00 = kernel[0],
                     b10 = kernel[1], b01 = kernel[2],
                     b20 = kernel[3], b02 = kernel[4],
                     b11 = kernel[5];

            reduce_t w22 = static_cast<reduce_t>(*wgt);
            auto wget = [&](offset_t o)
            {
                return bound::cget<reduce_t>(wgt, o);
            };

            reduce_t w12 = wget(ix1);
            reduce_t w32 = wget(ix3);
            reduce_t w21 = wget(iy1);
            reduce_t w23 = wget(iy3);

            reduce_t w02 = wget(ix0);
            reduce_t w42 = wget(ix4);
            reduce_t w20 = wget(iy0);
            reduce_t w24 = wget(iy4);

            reduce_t w11 = wget(ix1+iy1);
            reduce_t w13 = wget(ix1+iy3);
            reduce_t w31 = wget(ix3+iy1);
            reduce_t w33 = wget(ix3+iy3);

            reduce_t m12 = (b10 - 2*b20) * (w22 + w12)
                           - 2*b20 * (w32 + w02)
                           - b11 * (w21 + w11 + w23 + w13);
            reduce_t m32 = (b10 - 2*b20) * (w22 + w32)
                           - 2*b20 * (w42 + w12)
                           - b11 * (w23 + w33 + w21 + w31);

            reduce_t m21 = (b01 - 2*b02) * (w22 + w21)
                           - 2*b02 * (w23 + w20)
                           - b11 * (w12 + w11 + w32 + w13);
            reduce_t m23 = (b01 - 2*b02) * (w22 + w23)
                           - 2*b02 * (w24 + w21)
                           - b11 * (w32 + w33 + w12 + w31);

            reduce_t m02 = b20 * (2 * w12 + w02 + w22);
            reduce_t m42 = b20 * (2 * w32 + w42 + w22);
            reduce_t m20 = b02 * (2 * w21 + w20 + w22);
            reduce_t m24 = b02 * (2 * w23 + w24 + w22);
            reduce_t m11 = b11 * (w22 + w12 + w21 + w11);
            reduce_t m13 = b11 * (w22 + w12 + w23 + w13);
            reduce_t m31 = b11 * (w22 + w32 + w21 + w31);
            reduce_t m33 = b11 * (w22 + w32 + w23 + w33);

            b00 -= (m12*fx1 +  m32*fx3 +
                    m21*fy1 +  m23*fy3) +
                   (m02*fx0 +  m42*fx4 +
                    m20*fy0 +  m24*fy4) +
                   (m11*(fx1*fy1) +  m13*(fx1*fy3) +
                    m31*(fx3*fy1) +  m33*(fx3*fy3));

            op(*out, b00);
        }
    }

    //------------------------------------------------------------------
    //                         BENDING JRLS
    //------------------------------------------------------------------

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_bending_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t istride[D], const offset_t wstride[D],
        offset_t osc, offset_t isc, const reduce_t kernel[6*C])
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

        reduce_t w22 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o);
        };

        // first order neighbours
        reduce_t w12 = wget(wx1);
        reduce_t w32 = wget(wx3);
        reduce_t w21 = wget(wy1);
        reduce_t w23 = wget(wy3);

        // second order neighbours
        reduce_t w02 = wget(wx0);
        reduce_t w42 = wget(wx4);
        reduce_t w20 = wget(wy0);
        reduce_t w24 = wget(wy4);

        // diagonal neighbours
        reduce_t w11 = wget(wx1+wy1);
        reduce_t w13 = wget(wx1+wy3);
        reduce_t w31 = wget(wx3+wy1);
        reduce_t w33 = wget(wx3+wy3);

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t b00 = kernel[0],
                     b10 = kernel[1], b01 = kernel[2],
                     b20 = kernel[3], b02 = kernel[4],
                     b11 = kernel[5];

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            auto sum1 = [&]()
            {
                reduce_t m12 = (b10 - 2*b20) * (w22 + w12)
                               - 2*b20 * (w32 + w02)
                               - b11 * (w21 + w11 + w23 + w13);
                reduce_t m32 = (b10 - 2*b20) * (w22 + w32)
                               - 2*b20 * (w42 + w12)
                               - b11 * (w23 + w33 + w21 + w31);

                reduce_t m21 = (b01 - 2*b02) * (w22 + w21)
                               - 2*b02 * (w23 + w20)
                               - b11 * (w12 + w11 + w32 + w13);
                reduce_t m23 = (b01 - 2*b02) * (w22 + w23)
                               - 2*b02 * (w24 + w21)
                               - b11 * (w32 + w33 + w12 + w31);


                return (m12*get(ix1, fx1) +  m32*get(ix3, fx3) +
                        m21*get(iy1, fy1) +  m23*get(iy3, fy3));
            };

            auto sum2 = [&]()
            {
                reduce_t m02 = b20 * (2 * w12 + w02 + w22);
                reduce_t m42 = b20 * (2 * w32 + w42 + w22);
                reduce_t m20 = b02 * (2 * w21 + w20 + w22);
                reduce_t m24 = b02 * (2 * w23 + w24 + w22);

                return (m02*get(ix0, fx0) +  m42*get(ix4, fx4) +
                        m20*get(iy0, fy0) +  m24*get(iy4, fy4));
            };

            auto sumdiag = [&]()
            {
                reduce_t m11 = b11 * (w22 + w12 + w21 + w11);
                reduce_t m13 = b11 * (w22 + w12 + w23 + w13);
                reduce_t m31 = b11 * (w22 + w32 + w21 + w31);
                reduce_t m33 = b11 * (w22 + w32 + w23 + w33);

                return (m11*get(ix1+iy1, fx1*fy1) +  m13*get(ix1+iy3, fx1*fy3) +
                        m31*get(ix3+iy1, fx3*fy1) +  m33*get(ix3+iy3, fx3*fy3));
            };

            op(*out, b00*center + sum1() + sum2() + sumdiag());
        };

        for (offset_t c=0; c<C; ++c)
            conv(out + osc*c, inp + isc*c, kernel + 6*c);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_bending_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t wstride[D], offset_t osc,
        const reduce_t kernel[6*C])
    {
        offset_t   x = loc[0],       y = loc[1];
        offset_t  nx = size[0],     ny = size[1];
        offset_t wsx = wstride[0], wsy = wstride[1];

        signed char fx0 = bound_utils_x::sign(x-2, nx);
        signed char fx1 = bound_utils_x::sign(x-1, nx);
        signed char fx3 = bound_utils_x::sign(x+1, nx);
        signed char fx4 = bound_utils_x::sign(x+2, nx);
        signed char fy0 = bound_utils_y::sign(y-2, ny);
        signed char fy1 = bound_utils_y::sign(y-1, ny);
        signed char fy3 = bound_utils_y::sign(y+1, ny);
        signed char fy4 = bound_utils_y::sign(y+2, ny);
        offset_t    ix0 = (bound_utils_x::index(x-2, nx) - x) * wsx;
        offset_t    ix1 = (bound_utils_x::index(x-1, nx) - x) * wsx;
        offset_t    ix3 = (bound_utils_x::index(x+1, nx) - x) * wsx;
        offset_t    ix4 = (bound_utils_x::index(x+2, nx) - x) * wsx;
        offset_t    iy0 = (bound_utils_y::index(y-2, ny) - y) * wsy;
        offset_t    iy1 = (bound_utils_y::index(y-1, ny) - y) * wsy;
        offset_t    iy3 = (bound_utils_y::index(y+1, ny) - y) * wsy;
        offset_t    iy4 = (bound_utils_y::index(y+2, ny) - y) * wsy;

        reduce_t w22 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o);
        };

        reduce_t w12 = wget(ix1);
        reduce_t w32 = wget(ix3);
        reduce_t w21 = wget(iy1);
        reduce_t w23 = wget(iy3);

        reduce_t w02 = wget(ix0);
        reduce_t w42 = wget(ix4);
        reduce_t w20 = wget(iy0);
        reduce_t w24 = wget(iy4);

        reduce_t w11 = wget(ix1+iy1);
        reduce_t w13 = wget(ix1+iy3);
        reduce_t w31 = wget(ix3+iy1);
        reduce_t w33 = wget(ix3+iy3);

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t b00 = kernel[0],
                     b10 = kernel[1], b01 = kernel[2],
                     b20 = kernel[3], b02 = kernel[4],
                     b11 = kernel[5];

            reduce_t m12 = (b10 - 2*b20) * (w22 + w12)
                           - 2*b20 * (w32 + w02)
                           - b11 * (w21 + w11 + w23 + w13);
            reduce_t m32 = (b10 - 2*b20) * (w22 + w32)
                           - 2*b20 * (w42 + w12)
                           - b11 * (w23 + w33 + w21 + w31);

            reduce_t m21 = (b01 - 2*b02) * (w22 + w21)
                           - 2*b02 * (w23 + w20)
                           - b11 * (w12 + w11 + w32 + w13);
            reduce_t m23 = (b01 - 2*b02) * (w22 + w23)
                           - 2*b02 * (w24 + w21)
                           - b11 * (w32 + w33 + w12 + w31);

            reduce_t m02 = b20 * (2 * w12 + w02 + w22);
            reduce_t m42 = b20 * (2 * w32 + w42 + w22);
            reduce_t m20 = b02 * (2 * w21 + w20 + w22);
            reduce_t m24 = b02 * (2 * w23 + w24 + w22);

            reduce_t m11 = b11 * (w22 + w12 + w21 + w11);
            reduce_t m13 = b11 * (w22 + w12 + w23 + w13);
            reduce_t m31 = b11 * (w22 + w32 + w21 + w31);
            reduce_t m33 = b11 * (w22 + w32 + w23 + w33);

            b00 -= (m12*fx1 +  m32*fx3 +
                    m21*fy1 +  m23*fy3) +
                   (m02*fx0 +  m42*fx4 +
                    m20*fy0 +  m24*fy4) +
                   (m11*(fx1*fy1) +  m13*(fx1*fy3) +
                    m31*(fx3*fy1) +  m33*(fx3*fy3));

            op(*out, b00);
        };

        for (offset_t c=0; c<C; ++c)
            conv(out + osc*c, kernel + 6*c);
    }
};

} // namespace reg_field
} // namespace jf

#endif // JF_REGULARISERS_FIELD_2D
