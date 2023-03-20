#ifndef JF_REGULARISERS_FLOW_1D
#define JF_REGULARISERS_FLOW_1D
#include "../../cuda_switch.h"
#include "../../bounds.h"
#include "../../utils.h"
#include "utils.h"

namespace jf {
namespace reg_flow {

//----------------------------------------------------------------------
//          low-level kernels for anything regularization
//----------------------------------------------------------------------

template <typename scalar_t, typename reduce_t, typename offset_t,
          bound::type BX>
struct RegFlow<two, scalar_t, reduce_t, offset_t, BX> {
    using bound_utils_x = bound::utils<BX>;
    typedef scalar_t & (*OpType)(scalar_t &, const reduce_t &);

    //------------------------------------------------------------------
    //                            ABSOLUTE
    //------------------------------------------------------------------

    static const int kernelsize_absolute = 1;

    /// kernel <- [absx]
    __device__ static inline void
    make_kernel_absolute(
        reduce_t * kernel, reduce_t absolute, const reduce_t voxel_size[1])
    {
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);
        kernel[0] = absolute / vx;
    }

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_absolute(
        scalar_t * out, const scalar_t * inp,
        offset_t osc, offset_t isc, const reduce_t kernel[1])
    {
        op(out[0], kernel[0] * inp[0]);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline  void
    kernel_absolute(scalar_t * out, offset_t osc, const reduce_t kernel[1])
    {
        op(out[0], kernel[0]);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline  void
    diag_absolute(scalar_t * out, offset_t osc, const reduce_t kernel[1])
    {
        return kernel_absolute(out, osc, kernel);
    }

    //------------------------------------------------------------------
    //                            MEMBRANE
    //------------------------------------------------------------------

    static const int kernelsize_membrane = 2;

    /// kernel <- [absx, wx1]
    __device__ static inline void
    make_kernel_membrane(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[3])
    {
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);
        kernel[0]  = absolute / vx;
        kernel[1]  = -membrane;
    }

    /// kernel <- [wx0, wx1]
    __device__ static inline void
    make_fullkernel_membrane(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[2])
    {
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);
        kernel[0]  = (absolute + 2 * membrane * vx) / vx;
        kernel[1]  = -membrane;
    }

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_membrane(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[1], const offset_t size[1], const offset_t stride[1],
        offset_t osc, offset_t isc, const reduce_t kernel[2])
    {
        offset_t  x = loc[0];
        offset_t nx = size[0];
        offset_t sx = stride[0];

        offset_t x0 = x-1, x1 = x+1;
        signed char fx0 = bound_utils_x::sign(x0, nx);
        signed char fx1 = bound_utils_x::sign(x1, nx);
        x0 = (bound_utils_x::index(x0, nx) - x) * sx;
        x1 = (bound_utils_x::index(x1, nx) - x) * sx;

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t center = static_cast<reduce_t>(inp[0]);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out, kernel[0] * center +
                     kernel[1] * (get(x0, fx0) + get(x1, fx1));
        };

        conv(out, inp, kernel);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
    kernel_membrane(
        scalar_t * out, offset_t sc, const offset_t stride[1],
        const reduce_t kernel[2])
    {
        offset_t sx = stride[0];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0], w100 = kernel[1];
            op(out[0],   w000);
            op(out[-sx], w100);
            op(out[+sx], w100);
        };

        setkernel(out, kernel);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_membrane(
        scalar_t * out, offset_t osc,
         const offset_t loc[1], const offset_t size[1],
         const reduce_t kernel[2])
    {
        offset_t  x = loc[0];
        offset_t nx = size[0];

        signed char fx = bound_utils_x::sign(x-1, nx)
                       + bound_utils_x::sign(x+1, nx);

         op(out[0], kernel[0] - kernel[1]*fx);
    }

    //------------------------------------------------------------------
    //                            BENDING
    //------------------------------------------------------------------

    static const int kernelsize_bending = 3;

    /// kernel <- [absx, wx100, wx200]
    __device__ static inline void
    make_kernel_bending(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        const reduce_t voxel_size[1])
    {
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);

        reduce_t w100 = (-4 * bending * vx - membrane) * vx;
        reduce_t w200 = bending * vx * vx;
        reduce_t w000 = absolute;

        kernel[0]  = w000 / vx;
        kernel[1]  = w100 / vx;
        kernel[2]  = w200 / vx;
    }

    /// kernel <- [wx000, wx100, wx200]
    __device__ static inline void
    make_fullkernel_bending(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        const reduce_t voxel_size[1])
    {
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);

        reduce_t w100 = (-4 * bending * vx - membrane) * vx;
        reduce_t w200 = bending * vx * vx;
        reduce_t w000 = absolute - 2 * (w100 + w200);

        kernel[0]  = w000 / vx;
        kernel[1]  = w100 / vx;
        kernel[2]  = w200 / vx;
    }

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_bending(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[1], const offset_t size[1],
        const offset_t stride[1], offset_t osc, offset_t isc,
        const reduce_t kernel[3])
    {
        offset_t  x = loc[0];
        offset_t nx = size[0];
        offset_t sx = stride[0];

        signed char fx00 = bound_utils_x::sign(x-2, nx);
        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        signed char fx11 = bound_utils_x::sign(x+2, nx);
        offset_t    x00 = (bound_utils_x::index(x-2, nx) - x) * sx;
        offset_t    x0  = (bound_utils_x::index(x-1, nx) - x) * sx;
        offset_t    x1  = (bound_utils_x::index(x+1, nx) - x) * sx;
        offset_t    x11 = (bound_utils_x::index(x+2, nx) - x) * sx;

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1],
                     w200 = kernel[2];

            reduce_t center = static_cast<reduce_t>(inp[0]);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
                  w000 * center
                + w100 * (get(x0, fx0) + get(x1, fx1))
                + w200 * (get(x00, fx00) + get(x11, fx11))
            );
        };

        conv(out, inp, kernel);
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
     kernel_bending(
        scalar_t * out, offset_t sc, const offset_t stride[1],
        const reduce_t kernel[3])
    {
        offset_t sx = stride[0];

        auto setkernel = [&](scalar_t * o, const reduce_t * ker) {
            reduce_t w000 = ker[0],
                     w100 = ker[1],
                     w200 = ker[2];
            op(o[0],      w000);
            op(o[-sx],    w100);
            op(o[+sx],    w100);
            op(o[-sx*2],  w200);
            op(o[+sx*2],  w200);
        };

        setkernel(out, kernel);
    }

    // --- diagonal ---

    template <OpType op = set>
    static inline __device__ void
    diag_bending(
        scalar_t * out, offset_t osc,
        const offset_t loc[1], const offset_t size[1],
        const reduce_t kernel[3])
    {
        offset_t  x = loc[0];
        offset_t nx = size[0];

        signed char fx00 = bound_utils_x::sign(x-2, nx);
        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        signed char fx11 = bound_utils_x::sign(x+2, nx);

        auto setdiag = [&](scalar_t & out, const reduce_t * kernel) {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1]
                     w200 = kernel[2];
            w000 -= w100 * (fx0 + fx1) + w200 * (fx00 + fx11);
            op(out, w000);
        };
        setdiag(out[0], kernel);
    }

    //------------------------------------------------------------------
    //                          LAME + BENDING
    //------------------------------------------------------------------

    static const int kernelsize_all = 3;

    /// kernel <- [absx, wx100, wx200]
    static inline __device__ void
    make_kernel_all(
        reduce_t * kernel,
        reduce_t absolute, reduce_t membrane, reduce_t bending,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[1])
    {
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);

        reduce_t w100 = (-4 * bending * vx - membrane) * vx;
        reduce_t w200 = bending * vx * vx;
        reduce_t  w000 = absolute;

        kernel[0]  = w000 / vx;
        kernel[1]  = w100 / vx - 2*shears - div;
        kernel[2]  = w200 / vx;
    }

    /// kernel <- [wx000, wx100, wx200]
    static inline __device__ void make_fullkernel_all(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        reduce_t shears, reduce_t div,
        const reduce_t voxel_size[1])
    {
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);

        reduce_t w100 = (-4 * bending * vx - membrane) * vx;
        reduce_t w200 = bending * vx * vx;
        reduce_t w000 = absolute - 2 * (w100 + w200);

        kernel[0]  = w000 / vx + 4*shears + 2*div;
        kernel[1]  = w100 / vx - 2*shears - div;
        kernel[2]  = w200 / vx;
    }

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_all(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[1], const offset_t size[1],
        const offset_t stride[1], offset_t osc, offset_t isc,
        const reduce_t kernel[3])
    {
        offset_t  x = loc[0];
        offset_t nx = size[0];
        offset_t sx = stride[0];

        signed char fx00 = bound_utils_x::sign(x-2, nx);
        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        signed char fx11 = bound_utils_x::sign(x+2, nx);
        offset_t    x00 = (bound_utils_x::index(x-2, nx) - x) * sx;
        offset_t    x0  = (bound_utils_x::index(x-1, nx) - x) * sx;
        offset_t    x1  = (bound_utils_x::index(x+1, nx) - x) * sx;
        offset_t    x11 = (bound_utils_x::index(x+2, nx) - x) * sx;

        reduce_t center0 = static_cast<reduce_t>(inp[0]);

        auto cget0 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp, o, f) - center0;
        };
        auto get0 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp, o, f);
        };

        {
            reduce_t wx000 = kernel[0], wx100 = kernel[1], wx200 = kernel[2];

            op(out[0],
                  wx000 * center0
                + wx100 * (cget0(x0, fx0)   + cget0(x1, fx1))
                + wx200 * (cget0(x00, fx00) + cget0(x11, fx11))
            );
        }
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
    kernel_all(
        scalar_t * out, const offset_t sc[2],
        const offset_t stride[1], const reduce_t kernel[3])
    {
        offset_t sx = stride[0];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0], w100 = kernel[1], w200 = kernel[2];
            op(out[0],      w000);
            op(out[-sx],    w100);
            op(out[+sx],    w100);
            op(out[-sx*2],  w200);
            op(out[+sx*2],  w200);
        };

        setkernel(out, kernel);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline  void
    diag_all(
        scalar_t * out, offset_t osc,
        const offset_t loc[1], const offset_t size[1],
        const reduce_t kernel[3])
    {
        offset_t  x = loc[0];
        offset_t nx = size[0];

        signed char fx00 = bound_utils_x::sign(x-2, nx);
        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        signed char fx11 = bound_utils_x::sign(x+2, nx);

        auto setdiag = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0], w100 = kernel[1], w200 = kernel[2];
            w000 -= w100 * (fx0 + fx1) + w200 * (fx00 + fx11);
            op(*out, w000);
        };

        setdiag(out, kernel);
    }

    //------------------------------------------------------------------
    //                          LAME
    //------------------------------------------------------------------

    static const int kernelsize_lame = 2;

    /// kernel <- [absx, wx100]
    __device__ static inline  void
    make_kernel_lame(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[1])
    {
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);

        kernel[0] = absolute / vx;
        kernel[1] = -membrane - 2*shears - div;
    }

    /// kernel <- [wx000, wx100]
    __device__ static inline  void
    make_fullkernel_lame(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[1])
    {
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);

        reduce_t w100 = - membrane * vx;
        reduce_t w000 = absolute - 2 * w100;

        kernel[1] = - membrane - 2*shears - div;
        kernel[0] = absolute - 2*kernel[1];
    }

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_lame(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[1], const offset_t size[1],
        const offset_t stride[1], offset_t osc, offset_t isc,
        const reduce_t kernel[2])
    {
        offset_t  x = loc[0];
        offset_t nx = size[0];
        offset_t sx = stride[0];

        signed char fx0  = bound_utils_x::sign(x-1, nx);
        signed char fx1  = bound_utils_x::sign(x+1, nx);
        offset_t    x0  = (bound_utils_x::index(x-1, nx) - x) * sx;
        offset_t    x1  = (bound_utils_x::index(x+1, nx) - x) * sx;

        reduce_t wx000 = kernel[0], wx100 = kernel[1];

        reduce_t center0 = static_cast<reduce_t>(inp[0]);

        auto cget0 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp, o, f) - center0;
        };
        auto get0 = [&](offset_t o, signed char f)
        {
            return bound::cget<reduce_t>(inp, o, f);
        };

        op(out[0],
              wx000 * center0
            + wx100 * (cget0(x0, fx0) + cget0(x1, fx1))
        );
    }

    // --- kernel ---

    template <OpType op = set>
    __device__ static inline void
     kernel_lame(
        scalar_t * out, const offset_t sc[2], const offset_t stride[1],
        const reduce_t kernel[2])
    {
        const offset_t sx = stride[0];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0], w100 = kernel[1];
            op(out[0],      w000);
            op(out[-sx],    w100);
            op(out[+sx],    w100);
        };

        setkernel(out, kernel);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline  void
    diag_lame(
        scalar_t * out, offset_t osc,
        const offset_t loc[1], const offset_t size[1],
        const reduce_t kernel[2])
    {
        offset_t  x = loc[0];
        offset_t nx = size[0];

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);

        auto setdiag = [&](scalar_t & out, const reduce_t * kernel) {
            reduce_t w000 = kernel[0], w100 = kernel[1];
            w000 -= w100 * (fx0+fx1);
            op(out, w000);
        };

        setdiag(out[0], kernel);
    }

    //------------------------------------------------------------------
    //                         ABSOLUTE JRLS
    //------------------------------------------------------------------

    // --- matvec ---

    template <OpType op = set>
    static inline __device__
    void matvec_absolute_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t osc, offset_t isc, const reduce_t kernel[1])
    {
        reduce_t w = static_cast<reduce_t>(*wgt);
        op(out[0], kernel[0] * w * static_cast<reduce_t>(inp[0]));
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_absolute_jrls(
        scalar_t * out, const scalar_t * wgt,
        offset_t osc, const reduce_t kernel[1])
    {
        reduce_t w = static_cast<reduce_t>(*wgt);
        op(out[0], kernel[0] * w);
    }

    //------------------------------------------------------------------
    //                         MEMBRANE JRLS
    //------------------------------------------------------------------

    static const int kernelsize_membrane_jrls = kernelsize_membrane;

    __device__ static inline void
    make_kernel_membrane_jrls(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[1])
    {
        make_kernel_membrane(kernel, absolute, membrane, voxel_size);
        for (int k=0; k<kernelsize_membrane_jrls; ++k)
            kernel[k] *= 0.5;
    }

    // --- matvec ---

    template <OpType op = set>
    __device__ static inline void
    matvec_membrane_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[1], const offset_t size[1],
        const offset_t istride[1], const offset_t wstride[1],
        offset_t osc, offset_t isc, const reduce_t kernel[2])
    {
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t isx = istride[0];
        offset_t wsx = wstride[0];

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x);
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x);
        offset_t    wx0 = ix0 * wsx;
        offset_t    wx1 = ix1 * wsx;
        ix0 *= isx;
        ix1 *= isx;

        // --- load weight map ---

        reduce_t w111 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o) + w111;
        };
        reduce_t w011 = wget(wx0);
        reduce_t w211 = wget(wx1);

        // --- convolution ---

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t m000 = kernel[0], m100 = kernel[1];

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, signed char f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
               (m000*w111*2)*center
               + (m100*w011)*get(ix0, fx0) + (m100*w211)*get(ix1, fx1)
            );
        };

        conv(out, inp, kernel);
    }

    // --- diagonal ---

    template <OpType op = set>
    __device__ static inline void
    diag_membrane_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[1], const offset_t size[1],
        const offset_t wstride[1], offset_t osc,
        const reduce_t kernel[2])
    {
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t wsx = wstride[0];

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x) * wsx;
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x) * wsx;

        // --- load weight map ---

        reduce_t w111 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o) + w111;
        };
        reduce_t w011 = wget(ix0) * fx0;
        reduce_t w211 = wget(ix1) * fx1;

        // --- convolution ---

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t m000 = kernel[0], m100 = kernel[1];
            op(*out, m000*w111*2 - m100*(w011 + w211));
        };

        conv(out, kernel);
    }

    //------------------------------------------------------------------
    //                           LAME JRLS
    //------------------------------------------------------------------

    static const int kernelsize_lame_jrls = 2;

    /* kernel = [wx000, wx100]
     *
     * wx100 = -(0.5*div + shears)
     */
    __device__ static inline void
    make_kernel_lame_jrls(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[2])
    {
        make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

        for (int k=0; k < kernelsize_lame; ++k)
            kernel[k] *= 0.5;
    }

    // --- matvec ---

    template <OpType op = set>
    static inline __device__
    void matvec_lame_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[1], const offset_t size[1],
        const offset_t istride[1], const offset_t wstride[1],
        offset_t osc, offset_t isc, const reduce_t kernel[4])
    {
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t isx = istride[0];
        offset_t wsx = wstride[0];

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x);
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x);
        offset_t    wx0 = ix0 * wsx;
        offset_t    wx1 = ix1 * wsx;
        ix0 *= isx;
        ix1 *= isx;

        // --- load weight map ---

        reduce_t w111 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o);
        };

        reduce_t w011 = wget(wx0);
        reduce_t w211 = wget(wx1);

        // --- weight map kernel

        reduce_t wx000 = kernel[0],  wx100 = kernel[1];

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

        op(out[0],
           (wx000*w111*2)*center0
           + (wx100*(w011+w111))*cget0(ix0, fx0)
           + (wx100*(w211+w111))*cget0(ix1, fx1)
        );

    }

    // --- diagonal ---

    template <OpType op = set>
    static inline __device__
    void diag_lame_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[1], const offset_t size[1],
        const offset_t wstride[1], offset_t osc, const reduce_t kernel[2])
    {
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t wsx = wstride[0];

        signed char fx0 = bound_utils_x::sign(x-1, nx);
        signed char fx1 = bound_utils_x::sign(x+1, nx);
        offset_t    ix0 = (bound_utils_x::index(x-1, nx) - x) * wsx;
        offset_t    ix1 = (bound_utils_x::index(x+1, nx) - x) * wsx;

        // --- load weight map ---

        reduce_t w0 = static_cast<reduce_t>(*wgt);
        auto wget = [&](offset_t o)
        {
            return bound::cget<reduce_t>(wgt, o) + w0;
        };

        reduce_t wx = wget(ix0) * fx0 + wget(ix1) * fx1;

        // --- compute convolution ---

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t m000 = kernel[0], m100 = kernel[1];
            op(*out, m000*w0*2 - m100*wx);
        };

        conv(out, kernel);
    }

};

} // namespace reg_flow
} // namespace jf

#endif // JF_REGULARISERS_FLOW_1D
