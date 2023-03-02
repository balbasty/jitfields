#ifndef JF_RESTRICT_LOOP
#define JF_RESTRICT_LOOP
#include "../lib/cuda_switch.h"
#include "../lib/restrict.h"
#include "../lib/batch.h"
#include "../lib/parallel.h"

namespace jf {
namespace restrict {

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

    offset_t nx = size_inp[ndim-1];
    offset_t sx = stride_inp[ndim-1];
    scalar_t zx = scale[0];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t x;
            offset_t batch_offset = index2offset_1d<ndim>(i, size_out, stride_inp, x);
            offset_t out_offset = index2offset<ndim>(i, size_out, stride_out);

            Multiscale<one, zero, IX, BX>::restrict(
                out + out_offset, inp + batch_offset,
                x, nx, sx, zx, shift);
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

    offset_t nx = size_inp[ndim-2];
    offset_t ny = size_inp[ndim-1];
    offset_t sx = stride_inp[ndim-2];
    offset_t sy = stride_inp[ndim-1];
    scalar_t zx = scale[0];
    scalar_t zy = scale[1];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t x, y;
            offset_t batch_offset = index2offset_2d<ndim>(i, size_out, stride_inp, x, y);
            offset_t out_offset = index2offset<ndim>(i, size_out, stride_out);

            Multiscale<two, zero, IX, BX, IY, BY>::restrict(
                out + out_offset, inp + batch_offset,
                x, nx, sx, zx, y, ny, sy, zy, shift);
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
    offset_t numel = prod<ndim>(size_out);

    offset_t nx = size_inp[ndim-3];
    offset_t ny = size_inp[ndim-2];
    offset_t nz = size_inp[ndim-1];
    offset_t sx = stride_inp[ndim-3];
    offset_t sy = stride_inp[ndim-2];
    offset_t sz = stride_inp[ndim-1];
    scalar_t zx = scale[0];
    scalar_t zy = scale[1];
    scalar_t zz = scale[2];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t x, y, z;
            offset_t batch_offset = index2offset_3d(i, size_out, stride_inp, x, y, z);
            offset_t out_offset = index2offset(i, size_out, stride_out);

            Multiscale<three, zero, IX, BX, IY, BY, IZ, BZ>::restrict(
                out + out_offset, inp + batch_offset,
                x, nx, sx, zx, y, ny, sy, zy, z, nz, sz, zz, shift);
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

            Multiscale<D>::restrict(
                out + out_offset, inp + batch_offset,
                x, size_inp + ndim - D, stride_inp + ndim - D,
                corder, cbnd, scale, shift);
        }
    });
}

/* Special cases when scaling factor is bounded by (1, 2] */

template <int ndim, spline::type IX, bound::type BX,
          typename scalar_t, typename offset_t>
void loop21d(scalar_t * out, const scalar_t * inp,
             scalar_t shift, const scalar_t * scale,
             const offset_t * size_out,
             const offset_t * size_inp,
             const offset_t * stride_out,
             const offset_t * stride_inp)
{
    offset_t numel = prod<ndim>(size_out);

    offset_t nx = size_inp[ndim-1];
    offset_t sx = stride_inp[ndim-1];
    scalar_t zx = scale[0];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t x;
            offset_t batch_offset = index2offset_1d<ndim>(i, size_out, stride_inp, x);
            offset_t out_offset = index2offset<ndim>(i, size_out, stride_out);

            Multiscale<one, two, IX, BX>::restrict(
                out + out_offset, inp + batch_offset,
                x, nx, sx, zx, shift);
        }
    });
}

template <int ndim,
          spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          typename scalar_t, typename offset_t>
void loop22d(scalar_t * out, const scalar_t * inp,
             scalar_t shift, const scalar_t * scale,
             const offset_t * size_out,
             const offset_t * size_inp,
             const offset_t * stride_out,
             const offset_t * stride_inp)
{
    offset_t numel = prod<ndim>(size_out);

    offset_t nx = size_inp[ndim-2];
    offset_t ny = size_inp[ndim-1];
    offset_t sx = stride_inp[ndim-2];
    offset_t sy = stride_inp[ndim-1];
    scalar_t zx = scale[0];
    scalar_t zy = scale[1];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t x, y;
            offset_t batch_offset = index2offset_2d<ndim>(i, size_out, stride_inp, x, y);
            offset_t out_offset = index2offset<ndim>(i, size_out, stride_out);

            Multiscale<two, two, IX, BX, IY, BY>::restrict(
                out + out_offset, inp + batch_offset,
                x, nx, sx, zx, y, ny, sy, zy, shift);
        }
    });
}

template <int ndim,
          spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          spline::type IZ, bound::type BZ,
          typename scalar_t, typename offset_t>
void loop23d(scalar_t * out, const scalar_t * inp,
             scalar_t shift, const scalar_t * scale,
             const offset_t * size_out,
             const offset_t * size_inp,
             const offset_t * stride_out,
             const offset_t * stride_inp)
{
    offset_t numel = prod<ndim>(size_out);

    offset_t nx = size_inp[ndim-3];
    offset_t ny = size_inp[ndim-2];
    offset_t nz = size_inp[ndim-1];
    offset_t sx = stride_inp[ndim-3];
    offset_t sy = stride_inp[ndim-2];
    offset_t sz = stride_inp[ndim-1];
    scalar_t zx = scale[0];
    scalar_t zy = scale[1];
    scalar_t zz = scale[2];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t x, y, z;
            offset_t batch_offset = index2offset_3d<ndim>(i, size_out, stride_inp, x, y, z);
            offset_t out_offset = index2offset<ndim>(i, size_out, stride_out);

            Multiscale<three, two, IX, BX, IY, BY, IZ, BZ>::restrict(
                out + out_offset, inp + batch_offset,
                x, nx, sx, zx, y, ny, sy, zy, z, nz, sz, zz, shift);
        }
    });
}


} // namespace restrict
} // namespace jf

#endif // JF_RESTRICT_LOOP
