#ifndef JF_RESIZE_LOOP
#define JF_RESIZE_LOOP
#include "../lib/cuda_switch.h"
#include "../lib/resize.h"
#include "../lib/batch.h"
#include "../lib/parallel.h"

namespace jf {
namespace resize {

template <int ndim, spline::type IX, bound::type BX,
          typename scalar_t, typename offset_t>
void loop1d(scalar_t * out, const scalar_t * inp,
            scalar_t shift, const scalar_t * scale,
            const offset_t * size_out,
            const offset_t * size_inp,
            const offset_t * stride_out,
            const offset_t * stride_inp)
{
    offset_t numel = prod<ndim>(size_out);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t x;
            offset_t batch_offset = index2offset_1d<ndim>(i, size_out, stride_inp, x);
            offset_t out_offset = index2offset<ndim>(i, size_out, stride_out);

            Multiscale<one, IX, BX>::resize(out + out_offset, inp + batch_offset,
                                            x, size_inp[ndim-1], stride_inp[ndim-1],
                                            scale[0], shift);
        }
    });
}

template <int ndim,
          spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          typename scalar_t, typename offset_t>
void loop2d(scalar_t * out, const scalar_t * inp,
            scalar_t shift, const scalar_t * scale,
            const offset_t * size_out,
            const offset_t * size_inp,
            const offset_t * stride_out,
            const offset_t * stride_inp)
{
    offset_t numel = prod<ndim>(size_out);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t x, y;
            offset_t batch_offset = index2offset_2d<ndim>(i, size_out, stride_inp, x, y);
            offset_t out_offset = index2offset<ndim>(i, size_out, stride_out);

            Multiscale<two, IX, BX, IY, BY>::resize(
                out + out_offset, inp + batch_offset,
                x, size_inp[ndim-2], stride_inp[ndim-2], scale[0],
                y, size_inp[ndim-1], stride_inp[ndim-1], scale[1],
                shift);
        }
    });
}

template <int ndim,
          spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ,
          typename scalar_t, typename offset_t>
void loop3d(scalar_t * out, const scalar_t * inp,
            scalar_t shift, const scalar_t * scale,
            const offset_t * size_out,
            const offset_t * size_inp,
            const offset_t * stride_out,
            const offset_t * stride_inp)
{
    offset_t numel = prod(size_out, ndim);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t x, y, z;
            offset_t batch_offset = index2offset_3d<ndim>(i, size_out, stride_inp, x, y, z);
            offset_t out_offset = index2offset<ndim>(i, size_out, stride_out);

            Multiscale<three, IX, BX, IY, BY, IZ, BZ>::resize(
                out + out_offset, inp + batch_offset,
                x, size_inp[ndim-3], stride_inp[ndim-3], scale[0],
                y, size_inp[ndim-2], stride_inp[ndim-2], scale[1],
                z, size_inp[ndim-1], stride_inp[ndim-1], scale[2],
                shift);
        }
    });
}

template <int D, int ndim, typename scalar_t, typename offset_t>
void loopnd(scalar_t * out, const scalar_t * inp,
            scalar_t shift, const scalar_t * scale,
            const unsigned char * order,
            const unsigned char * bnd,
            const offset_t * size_out,
            const offset_t * size_inp,
            const offset_t * stride_out,
            const offset_t * stride_inp)
{
    offset_t numel = prod<ndim>(size_out);
    const spline::type * corder = reinterpret_cast<const spline::type *>(order);
    const bound::type  * cbnd   = reinterpret_cast<const bound::type *>(bnd);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t x[D];
            offset_t batch_offset = index2offset_nd<ndim>(i, size_out, stride_inp, x, D);
            offset_t out_offset = index2offset<ndim>(i, size_out, stride_out);

            Multiscale<D>::resize(
                out + out_offset, inp + batch_offset,
                x, size_inp + ndim - D, stride_inp + ndim - D,
                corder, cbnd, scale, shift);
        }
    });
}

} // namespace resize
} // namespace jf

#endif // JF_RESIZE_LOOP
